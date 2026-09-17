---
layout: ../../../layouts/PostLayout.astro
category: inference
title: Mini-SGLang 解析（3）
description: 解析 Mini-SGLang 的请求调度、分块 Prefill、Decode 与重叠执行
date: 2026-9-16
---
# Mini-SGLang 解析（3）

## 1. 概述

在 [Mini-SGLang 解析（1）](/my-blog/posts/inference/minisgl_1/) 中，我们介绍了 `Req`、`Batch` 与 `Engine`，但没有完整展开调度器如何把外部请求变成模型能够执行的批次。这篇博客沿着调度循环继续分析：新请求如何进入等待队列，Prefill 和 Decode 如何选择，请求状态如何推进，以及 CPU 调度工作如何与 GPU 计算重叠。

本文依据本地 Mini-SGLang 源码，版本为 `d0c2ecefca4ade346fb11d0d4ef1191ae73efdbc`。下面的源码路径均相对于项目根目录。

| 文件 | 主要职责 |
| --- | --- |
| `python/minisgl/scheduler/scheduler.py` | 接收消息、选择批次、准备张量、执行前向并回收请求 |
| `python/minisgl/scheduler/prefill.py` | 管理等待请求、资源准入和分块 Prefill |
| `python/minisgl/scheduler/decode.py` | 管理生成中的请求并组织 Decode 批次 |
| `python/minisgl/scheduler/table.py` | 分配请求表行并保存设备侧 token |
| `python/minisgl/scheduler/cache.py` | 分配 KV Cache 页并维护前缀缓存 |
| `python/minisgl/scheduler/io.py` | 处理 Tokenizer 与多个 TP rank 之间的消息传递 |

一次迭代的主干可以概括为：

```text
接收消息
   │
   ├─ UserMsg  → PrefillManager.pending_list
   ├─ AbortMsg → 查找并释放请求
   └─ ExitMsg  → 退出
   │
   ▼
优先选择 Prefill，否则选择 Decode
   │
   ▼
填充批次 → 分配 KV 页 → 准备位置、映射和 Attention 元数据
   │
   ▼
Engine.forward_batch() → 采样下一 token
   │
   ├─ GPU token 写回 token_pool
   └─ CPU token 判断 EOS、返回结果、回收资源
```

## 2. 调度器持有哪些状态

### 2.1. 两条请求队列

调度器没有用一个统一队列保存所有请求，而是按阶段分别管理：

| 管理器 | 集合 | 请求所处状态 |
| --- | --- | --- |
| `PrefillManager` | `pending_list: List[PendingReq]` | 尚未完成输入处理 |
| `DecodeManager` | `running_reqs: Set[Req]` | 已完成 Prefill，可以继续逐 token 生成 |

`pending_list` 是列表，调度时按进入顺序扫描。`running_reqs` 是集合，构造批次时再按 `uid` 排序，确保各个 TP rank 使用稳定一致的请求顺序。

`PendingReq` 只保存 CPU 输入、采样参数和可选的分块进度：

```python
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
```

它还没有占用请求表行。只有通过资源准入后，`PrefillAdder` 才会创建真正的 `Req`，分配 `table_idx` 并锁定匹配到的缓存句柄。

### 2.2. TableManager 与 token_pool

`TableManager` 管理可复用的请求表行：

```python
self._free_slots = list(range(max_running_reqs))
self.page_table = page_table
self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

`allocate()` 从列表尾部弹出一个行号，`free()` 把行号放回。相同的 `table_idx` 同时定位 `token_pool` 和 `page_table`：前者保存 token ID，后者保存每个逻辑位置对应的 KV Cache 槽位。

这里只清点行号，没有在释放时清空整行。旧值可以保留，因为后续请求会在读取相关位置前写入自己的 token 和页表项。额外的 dummy 行由引擎管理，不在 `TableManager` 的可分配范围内。

### 2.3. 两条 CUDA Stream

引擎创建自己的执行流，调度器又创建一条流：

```python
self.stream = torch.cuda.Stream(device=self.device)
self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
torch.cuda.set_stream(self.stream)
```

调度器流用于准备映射、复制输入和组织元数据；引擎流用于模型前向与采样。重叠模式提交新批次前执行：

```python
self.engine.stream.wait_stream(self.stream)
```

这让引擎流等待调度器流中为该批次安排的工作，同时不用同步整个设备。

## 3. 消息如何进入调度器

### 3.1. UserMsg：检查长度后加入 Prefill

`_process_one_msg()` 收到 `UserMsg` 后，先用引擎的实际最大序列长度限制输出预算：

```python
max_output_len = self.engine.max_seq_len - len(msg.input_ids)
```

输入已经占满上下文时，请求被丢弃。用户设置的 `max_tokens` 超过剩余空间时，调度器会直接修改该请求的采样参数，将其缩短到 `max_output_len`。通过检查后，请求被包装成 `PendingReq` 并追加到 `pending_list`。

这一步不匹配前缀、不分配表行，也不分配 KV 页。资源分配延迟到请求真正被选入 Prefill 批次时进行。

### 3.2. BatchBackendMsg、AbortBackendMsg 与 ExitMsg

`BatchBackendMsg` 只是消息容器，调度器递归处理其中每一项。它不会把多个用户消息强制合并成同一个模型批次，真正的动态组批仍由当前资源和 token 预算决定。

取消请求时，调度器先查询 Prefill 队列，再查询 Decode 集合：

```python
req_to_free = self.prefill_manager.abort_req(msg.uid)
req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
```

尚未开始的 `PendingReq` 没有设备资源，移出列表即可。已经开始分块 Prefill 的请求带有 `chunked_req`，需要释放表行和缓存；Decode 请求也需要执行相同的资源清理。`ExitMsg` 则通过抛出 `KeyboardInterrupt` 退出循环。

### 3.3. 单卡与 TP 多 rank 的消息一致性

在线模式下，主 rank 从 Tokenizer 接收请求并返回结果。TP 大于 1 时，主 rank 把原始消息广播给其他 rank；非主 rank 不向 Tokenizer 返回采样结果。

非阻塞接收时，主 rank 还会通过 CPU 进程组广播本轮待处理消息数量，其他 rank 按相同数量读取广播队列。这样每个 rank 都执行相同的请求状态转换和批次组织，避免张量并行各卡进入不同的 collective 调用。

## 4. Prefill 调度与资源准入

### 4.1. 每轮的两个预算

`PrefillManager.schedule_next_batch()` 创建临时的 `PrefillAdder`：

```python
adder = PrefillAdder(
    token_budget=prefill_budget,
    reserved_size=self.decode_manager.inflight_tokens,
    cache_manager=self.cache_manager,
    table_manager=self.table_manager,
)
```

`token_budget` 来自 `max_extend_tokens`，限制本轮 Prefill 实际处理的输入 token 数。`reserved_size` 是为正在 Decode 的请求估算的未来缓存需求，用来避免新 Prefill 把缓存容量全部占满。

二者单位都是 token，但用途不同：

- `token_budget` 控制一次模型前向的工作量。
- `reserved_size` 参与资源准入，保护已有生成请求后续所需的缓存。

### 4.2. Decode 的预留量

`DecodeManager.inflight_tokens` 的计算为：

```python
tokens_reserved = (page_size - 1) * len(running_reqs)
return sum(req.remain_len for req in running_reqs) + tokens_reserved
```

`remain_len` 是每个请求剩余的最大生成长度。额外的 `page_size - 1` 用于覆盖分页带来的内部空闲槽位：每个请求最后一页最多有这么多未使用位置。

这是保守估算。请求可能提前遇到 EOS，实际不会用完全部输出预算；但调度器不能在准入时依赖这种可能性。

### 4.3. 匹配前缀与双重容量检查

新请求的 `_try_allocate_one()` 依次执行：

1. 检查是否还有请求表行。
2. 通过 `CacheManager.match_req()` 匹配可复用前缀。
3. 计算本次需要扩展的输入长度。
4. 把输入扩展长度与完整输出预算加入容量估算。
5. 锁定缓存句柄后再次检查容量。
6. 分配表行，并复制命中前缀的 token 与页表位置。

核心估算是：

```python
extend_len = req.input_len - cached_len
estimated_len = extend_len + req.output_len
```

第一次检查发生在锁定句柄之前。锁定可能使原本可淘汰的缓存页变成不可淘汰，因此可用容量会减少；第二次检查用于捕捉这个变化。第二次失败时会解锁句柄并返回失败。

匹配最长前缀时至少保留最后一个输入 token 未缓存，所以正常请求仍有内容需要执行前向，并能得到第一个输出 token。

### 4.4. 队头阻塞与 Prefill 优先

调度器按 `pending_list` 顺序尝试加入请求。一旦某个请求无法加入，就立即停止扫描：

```python
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        ...
    else:
        break
```

因此，后面较小的请求不会绕过当前无法准入的请求。这保持了简单的到达顺序，也可能产生队头阻塞。

全局批次选择同样很直接：

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

只要能组成 Prefill 批次，本轮就不会调度 Decode。源码中的 TODO 明确说明未来可以支持其他策略；当前实现是 Prefill 优先，不是两类请求混合批处理。

## 5. Chunked Prefill：长输入如何分块

### 5.1. 创建 ChunkedReq

请求剩余输入大于本轮 `token_budget` 时，`_add_one_req()` 只复制一段 token，并创建 `ChunkedReq`：

```python
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
CLS = ChunkedReq if is_chunked else Req
```

它的 `input_ids` 只包含截至当前块末尾的前缀，`cached_len` 指向块开始位置。因此 `extend_len` 恰好是本轮需要计算的块长度。

`ChunkedReq` 覆盖了两个行为：

```python
def append_host(self, next_token):
    raise NotImplementedError

@property
def can_decode(self) -> bool:
    return False
```

中间块的 logits 没有业务意义，因此不能把采样结果追加为输出，也不能进入 Decode 集合。

### 5.2. 分块状态如何延续

创建中间块后，原 `PendingReq` 保存 `chunked_req`，并被放回等待列表前部。引擎完成前向时，`complete_one()` 会使 `ChunkedReq.cached_len` 更新到本块末尾，并把 `device_len` 加一。

不过下一次 `_add_one_req()` 使用的是 `chunked_req.cached_len`，而不是多加一后的 `device_len`。这样本轮采样产生的虚拟下一位置不会成为真实输入，下一块从刚完成 KV 的输入位置继续。

缓存页仍与同一个 `table_idx` 和 `cache_handle` 关联。后续块不会重新分配请求表行，也不会重新执行前缀匹配。

### 5.3. 最后一块转为普通 Req

当剩余输入能够在当前预算内处理时，`CLS` 变为普通 `Req`。该请求完成 Prefill 后，`DecodeManager.filter_reqs()` 会发现 `can_decode=True`，于是把它加入生成集合。

从调度器视角看，长请求经历的是：

```text
PendingReq
  → ChunkedReq（第 1 块）
  → ChunkedReq（第 2 块）
  → ...
  → Req（最后一块并产生首个输出 token）
  → Decode
```

`reserved_size` 在同一 Prefill 批次中增加的是请求完整的剩余输入与输出预算，而不只是当前块大小。这防止本轮继续接纳其他请求时忽略该长请求后面的资源需求。

## 6. Decode 调度

### 6.1. running_reqs 的维护

每次前向后，调度器执行：

```python
self.decode_manager.filter_reqs(forward_input.batch.reqs)
```

实现会把已有集合与本批请求取并集，再只保留 `can_decode` 为真的请求。普通 Prefill 请求因此进入 Decode；达到长度上限的请求被移除；`ChunkedReq.can_decode` 恒为假，不会误入集合。

EOS 此时还没有参与过滤，因为 CPU 副本可能尚未完成。EOS 请求会在 `_process_last_data()` 中显式 `remove_req()`。重叠调度下，这两个时间点之间可能已经提交了下一批，后文会解释其影响。

### 6.2. 每轮 Decode 处理一个 token

Decode 批次包含所有当前 `running_reqs`：

```python
return Batch(
    reqs=sorted(self.running_reqs, key=lambda req: req.uid),
    phase="decode",
)
```

普通 Decode 状态下，每个请求的 `extend_len` 为 1：上一轮产生的 token 已写入 `token_pool`，但它的 KV 尚未计算。本轮读出该 token，计算一个新位置，再采样下一个 token。

若 CUDA Graph 支持的批量档位大于真实请求数，`GraphRunner.pad_batch()` 会添加 dummy 请求。模型可能按填充后的大小执行，但采样结果和业务状态只处理 `batch.reqs` 中的真实请求。

## 7. 从 Batch 到模型输入

### 7.1. `_prepare_batch()` 的顺序

选出批次后，调度器按下面的顺序准备：

```python
self.engine.graph_runner.pad_batch(batch)
self.cache_manager.allocate_paged(batch.reqs)
batch.positions = _make_positions(batch, self.device)
input_mapping = _make_input_tuple(batch, self.device)
write_mapping = _make_write_tuple(batch, self.device)
batch.out_loc = self.engine.page_table[input_mapping]
self.engine.attn_backend.prepare_metadata(batch)
```

先填充请求，是因为位置和 Attention 元数据可能需要按图批次大小准备。KV 页只为真实请求分配；dummy 请求使用引擎预留的页和表行。

随后创建三个关键映射：

| 数据 | 形状含义 | 用途 |
| --- | --- | --- |
| `positions` | 每个待计算 token 的逻辑位置 | RoPE 与 Attention |
| `input_mapping` | `(table_idx, position)` | 从 `token_pool` 读取输入，也从 `page_table` 查 `out_loc` |
| `write_mapping` | `(table_idx, next_position)` | 把采样 token 写回设备表 |

最后准备 Attention 后端元数据与批量采样参数，并一起保存为 `ForwardInput`。在重叠执行中保留这些映射，可以避免请求状态推进后重新计算时发生不一致。

### 7.2. positions 与 input_mapping

对每个填充后的请求，位置范围是：

```python
range(req.cached_len, req.device_len)
```

这些范围按请求顺序拼接。`input_mapping` 的第一维为每个位置重复相应的 `table_idx`，第二维就是位置本身。

假设两个请求分别需要位置 `[3, 4, 5]` 与 `[8, 9]`，表行是 7 和 2，则映射相当于：

```text
row      = [7, 7, 7, 2, 2]
position = [3, 4, 5, 8, 9]
```

用这对索引读取 `token_pool` 得到扁平模型输入；读取 `page_table` 则得到相同 token 对应的 KV 写入位置。

### 7.3. write_mapping 中的 -1

模型前向与采样结束后，需要把下一 token 写入 `token_pool[table_idx, device_len]`，供下一轮 Decode 使用。映射创建时写为：

```python
write_list = [req.device_len if req.can_decode else -1 for req in batch.reqs]
```

`ChunkedReq` 不能 Decode，所以位置为 `-1`。PyTorch 的负索引会落到该行最后一列，中间块无用的采样 token 因此被写到保留位置，不会覆盖真实输入。该值仍会由引擎计算出来，但结果处理阶段直接跳过。

## 8. 前向、写回与完成处理

### 8.1. `_forward()`

执行阶段先根据保存的映射读取输入：

```python
batch.input_ids = self.token_pool[input_mapping]
forward_output = self.engine.forward_batch(batch, sample_args)
self.token_pool[output_mapping] = forward_output.next_tokens_gpu
self.decode_manager.filter_reqs(batch.reqs)
```

`Engine.forward_batch()` 会执行模型、对真实请求调用 `complete_one()`，再采样下一 token。因此写回映射必须在状态推进前准备好；否则读取更新后的 `device_len` 会错过应写入的位置。

GPU 结果直接写回 `token_pool`，下一轮模型输入无需等待 CPU。`ForwardOutput` 同时包含异步复制的 CPU token 和完成事件，供之后组织用户响应。

### 8.2. `_process_last_data()`

消费结果前先等待复制事件：

```python
copy_done.synchronize()
```

随后对每个非分块请求执行：

1. 把 CPU token 追加到 `req.input_ids`。
2. 根据长度预算判断 `not req.can_decode`。
3. 在没有设置 `ignore_eos` 时检查 EOS。
4. 创建 `DetokenizeMsg`。
5. 请求结束时移出 Decode 并释放资源。
6. 未结束的 Prefill 请求将前缀登记到缓存。

结束状态随每个 token 返回给 Tokenizer，所以外部可以流式解码，不必等待整个序列完成。

### 8.3. 释放与缓存的关系

资源释放入口是：

```python
self.table_manager.free(req.table_idx)
self.cache_manager.cache_req(req, finished=True)
```

归还表行与处理 KV Cache 是两个动作。`cache_req()` 会把可缓存前缀插入前缀缓存，解除旧句柄的锁，并按请求是否结束决定哪些尾部页可立即回收。

结果处理被包在 `lazy_free_region()` 中，使一组请求的缓存变更延迟合并，避免逐个释放时重复维护空闲页结构。KV Cache 的具体页分配、Radix Tree 插入和淘汰过程见 [Mini-SGLang 解析（4）](/my-blog/posts/inference/minisgl_4/)。

## 9. 普通循环与重叠循环

### 9.1. normal_loop

关闭重叠调度时，一轮严格按以下顺序运行：

```text
接收消息 → 选择并准备批次 → 前向 → 等待 CPU 复制 → 处理结果
```

整个循环位于引擎 Stream 上下文中。若当前没有可运行请求，消息接收使用阻塞模式，并在等待前调用 `run_when_idle()` 检查缓存一致性。

### 9.2. overlap_loop

重叠模式把前一批结果延后到下一轮处理：

```python
forward_input = self._schedule_next_batch()
if forward_input is not None:
    with self.engine_stream_ctx:
        self.engine.stream.wait_stream(self.stream)
        ongoing_data = (forward_input, self._forward(forward_input))

self._process_last_data(last_data)
return ongoing_data
```

时间线可以表示为：

```text
调度器流：准备批次 N+1 ─────────── 处理批次 N 的 CPU 结果
引擎流：              前向批次 N+1 ──────────────────
```

这样 CPU 上的追加 token、EOS 判断、消息发送和缓存维护可以与下一批 GPU 计算重叠。

### 9.3. 为什么可能多执行一轮

`_forward()` 在 CPU 结果处理前就把请求加入或保留在 `running_reqs` 中。若上一批刚生成 EOS，调度器在安排下一批时还不知道这个事实，因此该请求可能已经进入下一轮 Decode。

源码通过 `finished_reqs` 处理这种情况：上一批发现请求完成并释放资源后，下一批结果到来时会跳过第二次释放。这个集合只保存最近一轮新完成的请求，足以覆盖当前的单批流水深度。

其结果是重叠模式可能为刚结束的请求额外计算一个 token。`finished_reqs` 只避免重复释放，并没有阻止 `_process_last_data()` 再次追加 token 或发送返回项。调用方收到 `finished=True` 后应停止消费该请求；调度器和 Tokenizer 本身没有在此处过滤多余返回。这里体现了用少量推测执行换取 CPU 与 GPU 重叠的设计。

### 9.4. 何时阻塞等待消息

重叠循环仅在以下条件全部满足时阻塞：

- 没有上一批结果等待处理。
- 没有可运行的 Prefill 请求。
- 没有可运行的 Decode 请求。

只要还有流水线工作，消息接收就是非阻塞的，避免网络读取阻止批次推进。

## 10. 一个请求的完整状态变化

假设请求输入长度为 10，未命中缓存，`max_tokens=3`，Prefill 预算足够：

| 时刻 | 所在集合 | `cached_len` | `device_len` | 说明 |
| --- | --- | ---: | ---: | --- |
| 收到消息 | `pending_list` | — | — | 仅有 `PendingReq`，尚无设备资源 |
| 组成 Prefill | 当前批次 | 0 | 10 | 分配表行，复制 10 个输入 token |
| Prefill 前向后 | `running_reqs` | 10 | 11 | 得到第 1 个输出 token 并写到位置 10 |
| 第一次 Decode 后 | `running_reqs` | 11 | 12 | 位置 10 的 token 获得 KV，生成第 2 个 token |
| 第二次 Decode 后 | 移出集合 | 12 | 13 | 生成第 3 个 token，长度预算耗尽 |
| 结果处理 | 已结束 | 12 | 13 | 返回 `finished=True`，归还表行并缓存前缀 |

最后一个生成 token 不需要再次送入模型，因此它没有对应 KV；这就是结束时 `cached_len` 可能比 `device_len` 小 1 的原因。

如果输入需要分两块，则第一块前向后的采样结果被忽略，请求仍留在 `pending_list`；第二块转成普通 `Req` 后产生的 token 才是第一个用户可见输出。

## 11. 调度策略的实现边界

当前实现刻意保持简单，阅读时需要注意以下边界：

- Prefill 固定优先于 Decode，没有延迟感知或公平性策略。
- Prefill 按等待列表顺序准入，无法跳过暂时放不下的队头请求。
- Decode 每轮包含所有可运行请求，没有按优先级拆分。
- 容量准入按最大输出长度保守预留，可能低估可接受的并发量。
- `TableManager` 只维护空闲行列表，不检查重复释放或重复分配。
- EOS 依赖 CPU 结果判断，重叠模式允许一轮推测执行。
- `max_extend_tokens` 限制 Prefill 工作量；Decode 批量大小主要由运行请求数和 CUDA Graph 档位决定。

这些选择让请求生命周期和资源约束容易跟踪，也明确指出了进一步实现连续批处理策略时可以扩展的位置。

## 12. 串起完整调度流程

调度器接到请求后先做上下文长度检查，把它放入 Prefill 等待列表。每轮优先从列表头部组建 Prefill 批次，并同时检查表行、KV Cache 容量、前缀命中和 token 预算。长输入被拆成多个 `ChunkedReq`，只有最后一块会进入 Decode。

批次确定后，调度器分配缓存页，生成逻辑位置与二维索引，从设备侧 `token_pool` 取出本轮输入，并让引擎完成模型前向和采样。下一 token 立即写回 GPU 表供后续 Decode 使用，CPU 副本则用于流式返回、EOS 判断和资源回收。

普通模式串行完成这些步骤；重叠模式先提交下一批计算，再处理上一批结果，以隐藏 CPU 调度开销。至此，[第 1 篇](/my-blog/posts/inference/minisgl_1/) 中 Engine 两侧缺失的调度链路就完整连接起来了：外部消息经过动态组批成为 `Batch`，模型输出又经过状态更新与回收变回流式响应。
