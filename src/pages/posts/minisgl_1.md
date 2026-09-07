---
layout: ../../layouts/PostLayout.astro
title: Mini-SGLang 解析（1）
description: 解析 Mini-SGLang 的推理基本流程
date: 2026-9-7
---
# Mini-SGLang 解析（1）

## 1. 概述

我接下来会详细分析Mini-SGLang的各个组件以及核心功能。解析流程如下图：

[![Mini-SGLang 学习方向与代码入口概览](/my-blog/images/Mini-SGLang/overview.png)](/my-blog/images/Mini-SGLang/overview.png)

这篇博客将解析 Mini-SGLang 的推理基本流程。其对应的代码在 core.py 以及 engine.py 当中。

## 2. core.py 解析

首先，是对于 core.py 的分析。

该文件中记载了许多重要的类，包含 SamplingParams，Req，Batch以及Context。此外，该文件中还有着设置与获取全局上下文（global context）的函数。

### 2.1. SamplingParams

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

#### 2.1.1. 参数

| 参数        | 默认值 | 作用                              |
| ----------- | ------ | --------------------------------- |
| temperature | 0.0    | 调整采样概率分布的尖锐程度        |
| top_k       | -1     | 限制参与采样的候选token数量       |
| top_p       | 1.0    | 按累计概率限制候选token集合       |
| ignore_eos  | False  | 是否忽略解释token，不因它而停止   |
| max_tokens  | 1024   | 最多生成多少个新token，不包含输入 |

##### 2.1.1.1. temperature

模型输出的事词表中每个token的logits，而温度采用的基本计算是

$$
P_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

其中 $z_i$ 是某个token的logit，而 $T$ 则是温度。

- 0 < T < 1：分布更尖锐，更偏向高分 token。
- T = 1：使用原本的 softmax 分布。
- T > 1：分布更平坦，低分 token 更容易被选中。
- T = 0：不能直接代入公式；本项目通过贪心路径或极小温度处理。

通过改变概率差距，可以提高次选token的出现可能

##### 2.1.1.2. top_k

例如候选概率为：

```text
A: 0.50
B: 0.25
C: 0.15
D: 0.10
```

设置 top_k=2，只在 A、B 中采样。它并不意味着一定选 A。

在当前实现中：

- top_k=1：仅保留最高分候选。
- top_k=-1：不限制。
- 更准确地说，代码把所有 top_k < 1 的值都转换成词表大小，表示不限制。

##### 2.1.1.3. top_p

仍使用上面的概率：

```text
top_p = 0.70

A 的累计概率：0.50
A+B 的累计概率：0.75
```

按通常的 nucleus sampling 定义，保留 A、B，再从中采样。

区别在于：

- top_k 限制候选数量。
- top_p 限制候选集合的累计概率质量。

top_p=1.0 表示不做此项过滤。项目在随机采样路径中，会把它限制到 [1e-6, 1.0]。

##### 2.1.1.4. ignore_eos

EOS 是模型用于表示序列结束的特殊 token。

调度器中的逻辑相当于：

```python
finished = not req.can_decode

if not req.sampling_params.ignore_eos:
    finished |= (next_token == self.eos_token_id)
```

所以：

- False：达到长度上限，或者生成 EOS，都可以结束。
- True：EOS 不再触发结束，但长度上限仍然生效。

但是忽略 EOS 不等于禁止生成 EOS。 它只改变停止判断，没有修改 EOS 的采样概率。其具体逻辑见 python/minisgl/scheduler/scheduler.py:139。

##### 2.1.1.5. max_tokens

该参数表示生成长度上限。例如：

```text
输入长度 = 100
max_tokens = 20
```

表示最多新增 20 个 token，总长度最多达到 120；其中包含 Prefill 结束后采样得到的第一个输出 token。

调度器还会根据模型最大序列长度缩减这个值：

```python
max_output_len = max_seq_len - input_len

if sampling_params.max_tokens > max_output_len:
    sampling_params.max_tokens = max_output_len
```

因此用户传入的是期望上限，实际还受模型上下文长度限制。此外，调度器使用它估算未来的 KV Cache 需求。把 max_tokens 设得很大，即使模型实际很早结束，也可能影响当前可接纳的请求数量。 这里是资源预算估算，不代表立即分配全部输出长度对应的缓存页。

#### 2.1.2. 方法

##### 2.1.2.1. is_greedy

即是否使用贪心生成。贪心生成就是每步选择 logit 最大的 token：

```python
next_token = torch.argmax(logits, dim=-1)
```

本项目的判定条件为：

```python
(temperature <= 0.0 or top_k == 1) and top_p == 1.0
```

例如：

|                 配置                 | is_greedy |
| :-----------------------------------: | --------- |
|               默认配置               | True      |
|  temperature=0.8, top_k=1, top_p=1.0  | True      |
| temperature=0.8, top_k=20, top_p=0.9 | False     |
|      temperature=0.0, top_p=0.9      | False     |

Sampler.prepare() (python/minisgl/engine/sample.py:53) 根据整个 batch 选择执行方式：

- 所有请求都是贪心配置：直接走 argmax，省去概率计算与随机采样。
- batch 中存在随机采样请求：统一建立温度、Top-K、Top-P 张量；其中贪心请求的温度被设为 1e-6。

因此，混合 batch 中的贪心请求采用极小温度近似，不能把它与独立 argmax 路径视为实现上完全相同，尤其是分数并列时。

### 2.2. Req

`SamplingParams` 描述如何采样，`Req` 则记录一个请求在推理过程中的状态。输入序列、缓存进度、生成长度上限，以及请求在共享表中的位置，都集中在这里。

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```

#### 2.2.1. 参数

| 字段                | 含义                                                                     |
| ------------------- | ------------------------------------------------------------------------ |
| `input_ids`       | CPU 上的 token 序列；初始化时是输入，随后追加已经返回到 CPU 的生成 token |
| `table_idx`       | 请求在`page_table` 和 `token_pool` 中占用的行号，释放后可以复用      |
| `cached_len`      | 已有 KV Cache 的前缀长度                                                 |
| `output_len`      | 构造请求时传入的输出长度预算，不是已经生成的数量                         |
| `uid`             | 请求标识，用于返回结果和定位请求                                         |
| `sampling_params` | 该请求的采样参数                                                         |
| `cache_handle`    | 前缀缓存句柄，用于关联缓存管理器中的匹配前缀                             |
| `device_len`      | 设备侧推理调度使用的逻辑序列长度                                         |
| `max_device_len`  | 初始化输入长度加输出预算，是该请求的长度上限                             |

前七项是创建 `Req` 时传入的参数，没有在类定义中设置默认值；`device_len` 和 `max_device_len` 则由 `__post_init__()` 计算。下面按照表格顺序分别说明。

##### 2.2.1.1. input_ids

`input_ids` 保存 token ID 序列，类型为 `torch.Tensor`，并且必须位于 CPU。

举个例子，假设输入经过分词后得到三个 token：

```python
input_ids = torch.tensor([101, 205, 309], dtype=torch.int32, device="cpu")
```

初始化时 `len(input_ids) = 3`。如果本轮生成 token `412`，调度器等待 CPU 结果可用后，通过 `append_host()` 将序列更新为 `[101, 205, 309, 412]`。因此，这个字段随着生成推进，也会包含输出 token。

##### 2.2.1.2. table_idx

`table_idx` 是请求在共享表中的行号，由 `TableManager.allocate()` 分配。`token_pool` 与 `page_table` 使用同一个行号，分别保存该请求的 token ID 和对应的 KV Cache 位置。

例如 `table_idx = 3`，要访问这个请求位置为 5 的 token，可以理解为：

```python
token_id = token_pool[3, 5]
kv_location = page_table[3, 5]
```

位置从 0 开始，因此这里访问的是第六个 token。`kv_location` 是缓存位置索引，真正的 K、V 数据保存在缓存池里。

请求结束后，该行号会归还给表管理器，之后可以分配给其他请求。因此，`table_idx` 适合定位当前占用的资源，而不能作为请求长期不变的业务标识。

##### 2.2.1.3. cached_len

`cached_len` 表示当前序列中，前多少个 token 已经具有可复用的 KV Cache。它描述一个连续前缀：位置区间为 `[0, cached_len)`。

例如输入长度为 100，命中了前 80 个 token 的缓存，则初始化时 `cached_len = 80`。本轮只需要处理位置 `[80, 100)`，不必重新计算前 80 个 token 的 KV。

如果没有命中缓存，初始值为 0。前向完成后，`complete_one()` 将它更新为本轮计算前的 `device_len`，表示本轮输入也已经计算完 KV。

这个数值统计的是有效 token 数，不是分配的缓存页数。缓存按页分配时可能存在尚未使用的槽位，不能将这些容量也计入 `cached_len`。

##### 2.2.1.4. output_len

`output_len` 是创建请求时使用的最大新增 token 数。`PendingReq.output_len` 返回 `sampling_params.max_tokens`，Prefill 管理器再把它传给 `Req`。

例如，普通请求的输入长度为 100，`output_len = 20`，初始化得到的总长度上限就是 120。这个预算包含 Prefill 后采样得到的第一个输出 token。

`output_len` 不会随着每次生成递减。剩余预算由 `remain_len = max_device_len - device_len` 计算；已经生成多少 token，也不能直接读取 `output_len` 得到。

此外，调度器在构造请求之前会按模型最大序列长度限制 `max_tokens`，因此这里拿到的是经过上下文长度约束后的预算。请求仍可能因为生成 EOS 而在用完预算前结束。

##### 2.2.1.5. uid

`uid` 是整数形式的请求标识，用于把推理结果对应回原请求。调度器构造返回消息时会携带 `req.uid`，取消请求时也可以通过 `uid` 查找对应请求。

例如，一个请求的 `uid = 42`、`table_idx = 3`，表示“请求 42 当前使用共享表的第 3 行”。即使第 3 行之后被其他请求复用，也不代表它还是请求 42。

`Req` 本身只是保存传入的 `uid`，并不生成编号或检查编号是否重复。编号的分配与管理由上游流程负责。

##### 2.2.1.6. sampling_params

`sampling_params` 保存该请求对应的 `SamplingParams` 对象，将前面介绍的温度、Top-K、Top-P 和停止相关设置与请求关联起来。

例如：

```python
sampling_params = SamplingParams(
    temperature=0.8,
    top_k=20,
    top_p=0.9,
    ignore_eos=False,
    max_tokens=20,
)
```

采样器准备批次时读取各请求的采样配置，调度器处理返回结果时读取 `ignore_eos`。因此，同一个批次中的不同请求可以携带不同的采样参数。

这里保存的是对象引用，`Req` 不会自动复制它。还要注意：`max_device_len` 在请求初始化时就已经计算完成，之后仅修改 `sampling_params.max_tokens`，不会自动重算这个长度上限。

##### 2.2.1.7. cache_handle

`cache_handle` 是前缀缓存系统返回的句柄，类型为 `BaseCacheHandle`。它提供匹配前缀的长度，以及获取对应缓存位置的接口，供缓存管理器继续管理这段前缀。

Prefill 管理器匹配到可复用前缀后，会锁定句柄，并把 `cache_handle.get_matched_indices()` 返回的位置写入请求的页表。例如命中了 80 个 token，就可以把这 80 个位置关联到已有 KV，而不必重新计算。

这个句柄也参与缓存生命周期管理：缓存管理器插入请求前缀时会解锁旧句柄；对于尚未结束的请求，还会更新 `req.cache_handle` 并锁定新句柄，保护仍在使用的前缀不被淘汰。

`cache_handle.cached_len` 与 `req.cached_len` 不一定始终相等。前者描述该句柄关联的缓存前缀，后者描述请求当前已经计算完 KV 的长度；前向刚完成、前缀尚未重新插入缓存时，请求进度可以领先于旧句柄。

##### 2.2.1.8. device_len

`device_len` 是在初始化时由 `len(input_ids)` 得到的派生字段，用于记录设备侧推理的逻辑序列长度，不是 GPU Tensor 本身。

例如初始化输入为 100 个 token，则 `device_len = 100`。本轮前向后，`complete_one()` 将它增加到 101，为本轮生成的 token 推进一个位置。下一轮 Decode 会使用这个新 token 继续计算。

由于 CPU 结果通过异步复制返回，`device_len` 可能已经推进，而 `input_ids` 尚未追加结果。调度器因此使用 `device_len` 准备设备侧位置与长度，而不是始终依赖 CPU 序列的即时长度。

##### 2.2.1.9. max_device_len

`max_device_len` 同样是派生字段，在初始化时计算：

```python
self.max_device_len = len(self.input_ids) + self.output_len
```

对于输入长度 100、输出预算 20 的普通请求，它的值为 120，并在后续生成中保持不变。`device_len` 每次增加，剩余预算就相应减少；当两者相等时，`remain_len = 0`，`can_decode` 返回 `False`。

它是长度判断的上限，不代表初始化时就分配了这么多 token 的 KV Cache。缓存管理器仍根据当前需要处理的长度按页分配缓存。

#### 2.2.2. 方法

##### 2.2.2.1. 初始化与对象比较

`__post_init__()` 在 dataclass 自动生成的初始化方法之后执行。这里要求至少有一个 token 尚未缓存，即 `cached_len < device_len`。与之对应，`scheduler/cache.py` 匹配前缀时最多使用输入的前 `input_len - 1` 个 token，保留最后一个 token 用于前向计算并得到下一 token 的预测。

`eq=False` 表示不生成逐字段比较的相等方法，普通 `Req` 实例沿用对象身份进行比较和哈希。`DecodeManager.running_reqs` 使用集合保存请求，因此即使两个请求的内容相同，也可以作为不同对象管理，同时避免自动比较 Tensor 字段。

##### 2.2.2.2. remain_len 与 extend_len

```python
@property
def remain_len(self) -> int:
    return self.max_device_len - self.device_len

@property
def extend_len(self) -> int:
    return self.device_len - self.cached_len
```

`remain_len` 是距离长度上限还剩多少 token；`extend_len` 是当前已有序列中，还有多少 token 需要在本轮计算 KV。

例如输入有 100 个 token，命中了前 80 个 token 的缓存，允许生成 20 个 token，则初始化后：

```text
cached_len     = 80
device_len     = 100
max_device_len = 120
remain_len     = 20
extend_len     = 20
```

此时只需处理位置 `[80, 100)` 的输入。两个长度恰好都是 20，但一个描述未来的生成预算，另一个描述本轮需要处理的输入数量。

##### 2.2.2.3. complete_one 与 append_host

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1

def append_host(self, next_token: torch.Tensor) -> None:
    self.input_ids = torch.cat([self.input_ids, next_token])
```

这两个方法分别推进设备侧的逻辑状态和 CPU 上的 token 序列。

`Engine.forward_batch()` 在模型前向之后调用 `complete_one()`，随后执行采样。该更新表示本轮输入已经计算了 KV，并为本轮生成的一个 token 推进逻辑长度；它本身不执行采样，也不复制 token 或分配缓存。新生成 token 的 KV 要在后续前向中计算，所以更新后 `device_len - cached_len = 1`。

采样结果异步复制到 CPU 后，调度器在 `_process_last_data()` 中等待复制事件完成，再调用 `append_host()`。因此，启用计算与调度重叠时，不能始终假设 `len(req.input_ids) == req.device_len`：CPU 序列可能暂时落后于设备侧逻辑进度。

沿用上面的例子，忽略提前遇到 EOS 的情况：

| 时刻                                       | `cached_len` | `device_len` | `remain_len` | `extend_len` |
| ------------------------------------------ | -------------- | -------------- | -------------- | -------------- |
| Prefill 前                                 | 80             | 100            | 20             | 20             |
| Prefill 前向后调用`complete_one()`       | 100            | 101            | 19             | 1              |
| 第一次 Decode 前向后调用`complete_one()` | 101            | 102            | 18             | 1              |
| 生成第 20 个输出 token 的这一轮更新后      | 119            | 120            | 0              | 1              |

最后一行中，最新生成的 token 尚无 KV，但请求已经达到长度上限，不需要再为它执行下一轮 Decode。

##### 2.2.2.4. can_decode 与请求结束

```python
@property
def can_decode(self) -> bool:
    return self.remain_len > 0
```

`can_decode` 检查长度预算，并用于在调度器收到 CPU 采样结果后结合 `ignore_eos` 判断是否结束请求，并移除请求、释放资源。

分块 Prefill 还有一个例外：`scheduler/prefill.py` 定义的 `ChunkedReq` 继承 `Req`，将 `can_decode` 固定为 `False`，并禁止调用 `append_host()`。中间输入块处理完成后仍需继续 Prefill，不能作为正常生成请求进入 Decode；调度器也会跳过这些块的 token 返回处理。

##### 2.2.2.5. __repr__

`__repr__()` 则输出请求类型、表行号和三个长度字段，便于观察缓存进度与长度上限，不会把整个 token 序列打印出来。

### 2.3. Batch

`Req` 描述单个请求，`Batch` 把本轮一起执行的请求组织起来，并携带模型前向需要的张量。

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
```

#### 2.3.1. 参数

创建时只传入 `reqs` 和 `phase`。`field(init=False)` 表示该字段不属于自动生成的构造函数参数；这里的张量字段没有默认值，必须由后续流程赋值后才能访问。

| 字段              | 内容与设置位置                                                       |
| ----------------- | -------------------------------------------------------------------- |
| `reqs`          | 真实请求列表                                                         |
| `phase`         | 本轮执行 Prefill 还是 Decode；`Literal` 是类型提示，不是运行时校验 |
| `padded_reqs`   | `GraphRunner.pad_batch()` 准备的请求列表，必要时加入占位请求       |
| `positions`     | 调度器按请求拼接的 token 位置，用于 RoPE 和索引映射                  |
| `input_ids`     | 调度器在`_forward()` 中从 GPU `token_pool` 取出的本轮输入 token  |
| `out_loc`       | 根据页表查到的本轮 KV 写入位置，不是生成 token 的返回位置            |
| `attn_metadata` | Attention 后端根据当前批次生成的执行元数据                           |

##### 2.3.1.1. reqs

`reqs` 保存本轮真实参与执行的 `Req` 对象列表。调度器用它组织请求，引擎根据它推进请求状态，返回结果时也按这个列表对应到各个请求。

例如 `reqs = [req_a, req_b]` 表示本轮有两个真实请求。列表保存的是请求对象的引用，前向后对 `req_a` 调用 `complete_one()`，更新的仍是该请求本身。

这里的顺序还决定了输入拼接和输出对应关系。它只包含真实请求；为 CUDA Graph 补充的占位请求放在 `padded_reqs` 中。

##### 2.3.1.2. phase

`phase` 表示本轮推理阶段，取值为 `"prefill"` 或 `"decode"`，由创建批次的调度逻辑传入。

例如 `Batch(reqs=[req_a, req_b], phase="prefill")` 表示一起处理这两个请求尚未缓存的输入；普通 Decode 批次则通常为每个请求处理一个新 token。设置这个字段本身不会执行模型，也不会自动准备输入张量。

当前调度器先尝试调度 Prefill，没有可执行的 Prefill 批次时再选择 Decode。一个 `Batch` 只有一种 `phase`，并不在这个结构中混合两种阶段。

##### 2.3.1.3. input_ids

`Batch.input_ids` 与 `Req.input_ids` 的含义不同：前者是本轮送入模型的设备侧输入，后者是单个请求在 CPU 上维护的序列。

调度器在 `_forward()` 中通过 `self.token_pool[input_mapping]` 获取本轮输入并赋给该字段。例如两个请求分别需要处理 3 个和 2 个 token，则本轮输入按请求顺序拼接为 5 个 token，而不是把两个请求的完整历史序列重新输入模型。

##### 2.3.1.4. positions

`positions` 保存本轮各个输入 token 在所属请求中的逻辑位置，从 0 开始编号。它与 `input_ids` 一一对应，用于 RoPE 和页表索引；不同请求的位置分别计算，不是整个批次统一连续编号。

调度器的 `_make_positions()` 为每个请求生成 `[cached_len, device_len)` 的位置，再将它们拼接。例如，不考虑占位请求：

```text
请求 A：cached_len = 2，device_len = 5 → positions = [2, 3, 4]
请求 B：cached_len = 0，device_len = 2 → positions = [0, 1]

batch.positions = [2, 3, 4, 0, 1]
```

此时 `batch.size` 为 2，但本轮输入包含 5 个 token。`input_ids` 和 `out_loc` 也按相同顺序对应这些位置。普通 Decode 请求每轮通常只有一个待计算 token，所以不填充时，输入 token 数才会与请求数一致。

##### 2.3.1.5. out_loc

`out_loc` 保存本轮各个 token 的 KV Cache 写入位置。它与 `positions` 的区别是：`positions` 描述请求内部的逻辑位置，`out_loc` 描述缓存池中的物理槽位。

例如，请求 A 的逻辑位置 2 可能映射到缓存槽位 35，而请求 B 的逻辑位置 0 可能映射到槽位 80。后端依照这些索引存储本轮的 K、V，不能把它理解成输出 token 的 ID。

页表查询可以概括为以下关系：

```python
batch.out_loc = page_table[request_row_indices, batch.positions]
```

其中行索引根据每个请求的 `table_idx` 重复相应次数。真实实现会准备索引张量及其类型；Attention 后端随后使用这些位置写入各层 KV Cache。

##### 2.3.1.6. padded_reqs

`padded_reqs` 是为本轮执行准备的请求列表，由 `GraphRunner.pad_batch()` 设置。它按原有顺序保留 `reqs`，必要时在末尾追加占位请求。

CUDA Graph 需要复用已经捕获的执行形状。对于能够使用 CUDA Graph 的 Decode 批次，`pad_batch()` 选择不小于真实请求数的最小可用捕获档位，并补上 `dummy_req`。例如真实请求数为 3、可用档位包含 4 时，`size` 为 3，`padded_size` 为 4。不使用 CUDA Graph 时，两者相等。

占位请求使用专门的表行和 dummy 缓存页。引擎采样时只使用 `logits[:batch.size]`，并只对 `batch.reqs` 更新请求状态，避免把填充部分当成真实输出。

##### 2.3.1.7. attn_metadata

`attn_metadata` 保存 Attention 后端为当前批次准备的元数据，类型为 `BaseAttnMetadata`。它在后端的 `prepare_metadata(batch)` 中设置，具体内容由所选后端决定。

例如，FlashAttention 后端会根据 `padded_reqs` 生成查询序列的累计长度、KV 序列长度和后端所需的页表。若两个请求本轮分别处理 3 个和 2 个 token，查询累计长度就是 `[0, 3, 5]`，后端由此区分拼接输入中各请求的边界。

这些元数据描述如何组织 Attention 计算，不是模型权重，也不是缓存中的 K、V 数据。创建 `Batch` 时该字段尚未赋值，必须先完成后端准备流程才能使用。

#### 2.3.2. 方法

`is_prefill`、`is_decode` 分别比较 `phase`；`size` 返回 `len(reqs)`，`padded_size` 返回 `len(padded_reqs)`。

##### 2.3.2.1. is_prefill

```python
@property
def is_prefill(self) -> bool:
    return self.phase == "prefill"
```

用于判断当前批次是否处于 Prefill 阶段。访问 `batch.is_prefill` 时返回布尔值，不会改变批次阶段或执行前向。

##### 2.3.2.2. is_decode

```python
@property
def is_decode(self) -> bool:
    return self.phase == "decode"
```

用于判断当前批次是否处于 Decode 阶段。CUDA Graph 的使用条件之一就是 `batch.is_decode` 为真，但还需要满足批次大小等其他条件。

##### 2.3.2.3. size

```python
@property
def size(self) -> int:
    return len(self.reqs)
```

返回真实请求数。例如两个请求分别输入 3 个和 2 个 token，`batch.size` 仍为 2，而不是输入 token 总数 5。引擎采样时用它截取真实请求对应的 logits。

##### 2.3.2.4. padded_size

```python
@property
def padded_size(self) -> int:
    return len(self.padded_reqs)
```

返回包含占位请求的执行规模。例如三个真实请求被补到四个请求的图捕获档位时，`batch.padded_size` 为 4。访问前必须先设置 `padded_reqs`，图回放使用这个大小选择对应的 CUDA Graph。

### 2.4. Context

`Context` 保存运行时共享资源，以及当前正在前向计算的批次。模型层可以通过它取得位置、缓存和后端，不必逐层传递这些参数。

```python
@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)
```

#### 2.4.1. 参数

| 字段             | 作用                                          |
| ---------------- | --------------------------------------------- |
| `page_size`    | KV Cache 分页使用的每页 token 数              |
| `page_table`   | 从请求行号和逻辑 token 位置映射到缓存物理位置 |
| `attn_backend` | 当前引擎使用的 Attention 后端                 |
| `moe_backend`  | MoE 执行后端；引擎只在 MoE 模型中设置         |
| `kv_cache`     | 各层 K、V 的缓存池                            |
| `_batch`       | 当前前向的批次，未进入前向上下文时为`None`  |

这些资源由 `Engine.__init__()` 分阶段初始化。`Context(page_size)` 本身不会创建缓存池或 Attention 后端，声明了字段也不代表字段已经可以使用。

这里特别容易误读的是 `page_table` 上的注释。它不表示实际缓存的 `page_size` 必须为 1，而是说该表按单个 token 的粒度记录原始缓存位置。KV Cache 仍可按多 token 的页分配，后端再按自己的需要处理这些索引。

`page_size` 是构造 `Context` 时唯一需要传入的参数；其余字段由引擎初始化或前向流程设置。下面分别说明它们保存什么，以及如何参与推理。

##### 2.4.1.1. page_size

`page_size` 表示每个 KV Cache 页能够容纳多少个 token 的缓存，由引擎配置传入。它的单位是 token，不是字节；实际显存占用还取决于模型层数、KV head 数量、head 维度和数据类型。

例如，假设 `page_size = 16`，一个没有可复用前缀的请求需要保存 20 个 token 的 KV，则需要两页，共提供 32 个 token 的槽位。其中只有 20 个位置有效，剩余位置是分页分配带来的空余容量。

`Context` 只保存这个配置值。具体的页分配由缓存管理器完成，缓存池和 Attention 后端也需要按相同的分页约定处理数据。

##### 2.4.1.2. page_table

`page_table` 是 GPU 上的二维整数张量。行对应请求占用的 `table_idx`，列对应请求内部的逻辑 token 位置，表项记录该 token 的 KV 在缓存池中的物理位置。

例如：

```python
location = ctx.page_table[req.table_idx, 5]
```

这里查询的是该请求第六个 token 的缓存位置。假设返回的位置为 35，且 `page_size = 16`，则按连续槽位编号理解，它对应编号为 2 的页、页内偏移为 3 的槽位。页号与偏移都从 0 开始。

因此，即使实际按 16 个 token 一页分配，表中仍然逐 token 记录位置。调度器使用这张表生成 `Batch.out_loc`，Attention 后端再据此访问缓存。

引擎创建的表有 `max_running_req + 1` 行，额外一行供占位请求使用；列数则将最大序列长度向上对齐到 32 的倍数。这个对齐是存储布局安排，不会相应增加请求允许使用的上下文长度。

##### 2.4.1.3. attn_backend

`attn_backend` 保存当前引擎使用的 Attention 执行后端，类型为 `BaseAttnBackend`。引擎根据配置和模型信息调用 `create_attention_backend()` 创建它。

不同后端对元数据与缓存布局的要求可能不同，因此调度器先调用 `prepare_metadata(batch)` 准备当前批次需要的信息，模型层再通过统一接口执行 Attention。

`AttentionLayer.forward()` 中的调用为：

```python
o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
```

其中 `q`、`k`、`v` 是本层产生的张量，`layer_id` 指定模型层，`ctx.batch` 提供位置、缓存写入索引和后端元数据。模型层由此可以调用所选后端，而不必自行实现每种后端的执行细节。

##### 2.4.1.4. moe_backend

`moe_backend` 保存混合专家模型的执行后端，类型为 `BaseMoeBackend`。模型中的 MoE 层通过 `ctx.moe_backend.forward()` 调用它，执行路由结果对应的专家计算。

引擎仅在模型配置表明它是 MoE 模型时初始化该字段：

```python
if config.model_config.is_moe:
    self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
```

例如运行普通稠密模型时，不需要走 MoE 专家计算，也就不会设置这个字段。这里的 `field(init=False)` 没有提供默认值，因此未初始化时不能把它当作已经存在、值为 `None` 的字段来读取。

`Context` 保存的是后端对象；具体调用所需的路由信息、隐藏状态和权重等参数，由 MoE 层准备并传给后端。

##### 2.4.1.5. kv_cache

`kv_cache` 保存 KV Cache 池对象，类型为 `BaseKVCachePool`。缓存池存放各层已经计算得到的 Key 和 Value，后续 token 做 Attention 时可以复用这些结果。

它与 `page_table` 分工不同：`page_table` 保存位置索引，`kv_cache` 保存这些位置对应的实际 K、V 数据。例如，Attention 后端通过下面的接口写入本轮计算结果：

```python
self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
```

其中 `batch.out_loc` 决定写入哪些槽位，`layer_id` 决定写入哪一层。请求的逻辑 token 位置可以在各层对应相同的槽位编号，但每一层保存的是自己的 K、V 张量。

引擎通过 `create_kvcache_pool()` 创建缓存池，并将同一个对象同时赋给 `self.kv_cache` 和 `self.ctx.kv_cache`，不会因此复制两份缓存。创建时还会额外预留一个 dummy 页，供占位请求使用。

##### 2.4.1.6. _batch

`_batch` 保存当前前向计算使用的 `Batch`，默认值为 `None`。它属于临时运行状态：进入 `forward_batch()` 时设置，离开时清空。

例如：

```python
with ctx.forward_batch(batch):
    positions = ctx.batch.positions
```

在这个代码块内部，`ctx.batch` 返回的就是当前传入的 `batch`，模型层可以通过它读取本轮位置与元数据。离开代码块后，`_batch` 恢复为 `None`；此时再读取 `ctx.batch` 会触发“没有活动批次”的断言。

下划线前缀表示它是内部状态，使用方通过 `batch` 属性读取、通过 `forward_batch()` 管理有效期。这个字段保存的是批次对象的引用，清空它不会删除请求，也不会释放请求占用的缓存页。

#### 2.4.2. 方法

##### 2.4.2.1. batch 与 forward_batch

```python
@property
def batch(self) -> Batch:
    assert self._batch is not None, "No active batch in context"
    return self._batch

@contextmanager
def forward_batch(self, batch: Batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

`batch` 属性要求当前存在活动批次。`forward_batch()` 则管理这个批次的有效期：进入 `with` 时设置 `_batch`，离开时清空；即使模型前向抛出异常，`finally` 也会执行清理。

引擎中的使用方式为：

```python
with self.ctx.forward_batch(batch):
    if self.graph_runner.can_use_cuda_graph(batch):
        logits = self.graph_runner.replay(batch)
    else:
        logits = self.model.forward()
```

在普通模型前向中，`AttentionLayer.forward()` 通过 `get_global_ctx()` 获取上下文，使用 `ctx.batch.positions` 执行 RoPE，并将 `ctx.batch` 传给 Attention 后端。

这里禁止嵌套进入 `forward_batch()`，避免一个前向覆盖另一个前向的活动批次。它只管理 `_batch` 的引用，不负责释放 KV Cache，也不是用于隔离多个线程或异步任务的上下文机制。

### 2.5. 全局上下文的设置与获取

```python
_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
```

`Engine` 创建 `Context` 后调用 `set_global_ctx()`，后续组件通过 `get_global_ctx()` 取得同一个对象。初始化阶段继续向该对象写入缓存池和后端，使用方获取到的仍是这一对象。

两个断言分别检查重复设置和未初始化访问。这里的“全局”是当前 Python 进程内的模块全局变量，不是多 GPU 进程之间共享的 Python 对象；各进程仍需初始化自己的上下文。该文件也没有提供重置函数。

需要区分两个生命周期：`_GLOBAL_CTX` 持续保存引擎共享资源，而 `Context._batch` 随每次前向临时设置和清空。退出 `forward_batch()` 只清空当前批次，全局上下文仍然存在。

### 2.6. 从这些结构串起一次推理

以普通、非分块请求为例，相关调用关系如下：

1. Prefill 管理器创建 `Req`，记录输入、命中的缓存长度和输出预算，并组织成 `Batch`。
2. 调度器准备填充请求、缓存位置、位置张量和 Attention 元数据，在执行前取得本轮 `input_ids`。
3. 引擎进入 `Context.forward_batch()`，执行模型前向或 CUDA Graph 回放。
4. 引擎调用 `Req.complete_one()` 推进逻辑长度，采样新 token，并发起到 CPU 的异步复制。
5. 调度器把 GPU 输出写回 `token_pool`，并按剩余长度维护 Decode 请求集合。
6. CPU 结果可用后，调度器调用 `append_host()`，判断长度上限与 EOS，返回结果并处理结束请求的资源。

启用重叠调度时，不同批次的上述步骤可以交错执行。这也解释了为什么 `Req` 要将设备侧长度更新与 CPU 序列追加拆开，以及为什么 `Batch` 需要显式保存本轮前向的输入与元数据。

本节对应的源码入口均位于 `python/minisgl/` 下：`core.py` 定义数据结构，`scheduler/prefill.py` 创建请求，`scheduler/scheduler.py` 准备批次并处理返回结果，`engine/engine.py` 执行前向，`engine/graph.py` 负责图执行所需的填充，`layers/attention.py` 展示模型层如何读取上下文。

## 3. engine.py 解析

前面的 `core.py` 定义请求、批次与共享上下文；`engine/engine.py` 则把它们与模型、缓存、通信和采样连接起来。`Engine` 接收调度器准备好的批次，完成一次前向并返回下一 token 的结果。

这里按照执行逻辑展开：先看初始化如何建立运行环境，再分别分析初始化调用的辅助方法、每轮前向以及资源清理。

### 3.1. Engine 的职责与调用关系

| 阶段       | 入口                                               | 主要作用                                           |
| ---------- | -------------------------------------------------- | -------------------------------------------------- |
| 启动       | `__init__()`                                     | 建立设备与通信环境，加载模型，分配缓存并准备图执行 |
| 通信初始化 | `_init_communication()`                          | 创建进程组，选择 TP 通信实现                       |
| 权重准备   | `_load_weight_state_dict()`                      | 生成测试权重或加载真实权重                         |
| 容量规划   | `_sync_get_memory()`、`_determine_num_pages()` | 汇总空闲显存并计算 KV Cache 页数                   |
| 批次执行   | `forward_batch()`                                | 执行模型、推进请求状态、采样并返回结果             |
| 退出       | `shutdown()`                                     | 按依赖顺序销毁图与通信资源                         |

请求选择、缓存页的动态分配、EOS 停止判断与结果发送由调度器负责。调用 `Engine.forward_batch()` 前，调度器已经准备了批次输入、位置、KV 写入位置、Attention 元数据和采样参数。

### 3.2. __init__：建立推理运行环境

#### 3.2.1. 检查初始状态并调整配置

```python
assert not torch.cuda.is_initialized()
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
_adjust_config(config)
```

构造函数首先要求当前进程尚未初始化 CUDA，然后注册 Tensor Parallel（TP，张量并行）的 rank 和规模。模型层构造及权重切分都会读取 TP 信息，因此这一步必须在模型创建之前完成。

`_adjust_config()` 将自动选择的后端和相应分页约束落实到配置中。后续 `Context`、缓存池和后端都使用调整后的配置，避免它们采用不同的 `page_size`。具体调整规则在后面的辅助函数小节中说明。

#### 3.2.2. 设置设备、执行流与上下文

```python
self.device = torch.device(f"cuda:{config.tp_info.rank}")
torch.cuda.set_device(self.device)
torch.manual_seed(42)
self.stream = torch.cuda.Stream()
torch.cuda.set_stream(self.stream)
self.dtype = config.dtype
self.ctx = Context(config.page_size)
set_global_ctx(self.ctx)
```

当前实现用 TP rank 直接选择进程可见的 CUDA 设备编号。例如 rank 为 1 时使用 `cuda:1`。随后设置随机种子、创建引擎执行流，并将它设置为当前 CUDA 流。

`self.stream` 是引擎执行模型与采样时使用的流。后面的 `forward_batch()` 会检查调用时的当前流是否与它一致，调度器需要在正确的流上下文中调用引擎。

`self.dtype` 决定权重和 KV Cache 使用的数据类型。`Context` 此时先创建并注册到全局，缓存池和后端字段在后续阶段补齐；注册上下文本身不表示全部资源已经就绪。

#### 3.2.3. 建立通信并测量初始显存

```python
self.tp_cpu_group = self._init_communication(config)
init_free_memory = self._sync_get_memory()[1]
```

通信组要先于显存汇总建立，因为 `_sync_get_memory()` 会通过 CPU 通信组在各个 TP rank 之间归约空闲显存。

该方法返回 `(最小空闲显存, 最大空闲显存)`，这里取下标 1，也就是各 rank 的最大值，作为模型加载前的显存基准。它已经位于通信初始化之后，因此后续通过两次测量的差值估算的是这一时点之后的显存变化。

#### 3.2.4. 创建模型结构并装载权重

```python
set_rope_device(self.device)
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)
self.model.load_state_dict(self._load_weight_state_dict(config))
```

`set_rope_device()` 设置 RoPE 使用的设备。模型结构则在 `meta` 设备上下文中创建，先描述张量形状和类型，再装入实际权重，避免先为占位权重分配一份真实存储。

`torch_dtype()` 是项目提供的上下文管理器，临时设置默认浮点类型，并在退出时恢复。这样创建出来的模型张量类型与后续加载权重的目标类型保持一致。

这里还需要结合 `layers/base.py` 理解：项目的模型使用自定义 `BaseOP.load_state_dict()`，它核对张量形状和类型后，通过 `setattr()` 将加载的真实张量放到对应属性中。因此这里可以把 meta 张量替换成设备上的实际权重。

#### 3.2.5. 计算容量并创建 KV Cache

```python
self.num_pages = self._determine_num_pages(init_free_memory, config)
num_tokens = self.num_pages * config.page_size
self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
    model_config=config.model_config,
    num_pages=self.num_pages + 1,
    page_size=config.page_size,
    device=self.device,
    dtype=self.dtype,
)
```

模型加载后再计算缓存容量，是为了把权重占用纳入预算。`self.num_pages` 是正常请求可使用的页数，`num_tokens` 是这些页能够容纳的 token 总数。

实际创建缓存池时额外增加一页，供 CUDA Graph 的占位请求使用。`self.kv_cache` 与 `self.ctx.kv_cache` 引用同一个缓存池，前者方便引擎访问，后者方便模型层和后端通过上下文访问。

这里完成缓存池的整体创建。具体请求使用哪些页，则由调度器侧的缓存管理器在运行时安排。

#### 3.2.6. 创建请求页表并限制最大长度

```python
self.max_seq_len = min(config.max_seq_len, num_tokens)
aligned_max_seq_len = _align_up_32(self.max_seq_len)
self.ctx.page_table = self.page_table = torch.zeros(
    (config.max_running_req + 1, aligned_max_seq_len),
    dtype=torch.int32,
    device=self.device,
)
```

单个请求的最大长度同时受到配置长度与缓存总容量限制。假设配置允许 8192 个 token，但正常缓存只有 4096 个槽位，`self.max_seq_len` 就会被限制为 4096。

页表每行对应一个请求，每列对应一个逻辑 token 位置。额外一行用于占位请求；列数向上对齐到 32 的倍数，每项为 4 字节的 `int32`，因此每行占用的字节数是 128 的倍数。

对齐后的列数只影响表的存储布局。例如逻辑上限为 1000 时分配 1024 列，允许的请求长度仍然是 1000。

#### 3.2.7. 创建计算后端与采样器

```python
self.ctx.attn_backend = self.attn_backend = create_attention_backend(
    config.attention_backend, config.model_config
)
if config.model_config.is_moe:
    self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
self.sampler = Sampler(self.device, config.model_config.vocab_size)
```

Attention 后端在缓存池和页表建立之后创建，因为后端初始化会读取这些资源。MoE 后端仅在 MoE 模型中创建。

采样器接收设备和词表大小，每轮根据 logits 与采样参数选择下一 token。初始化 `Sampler` 不会立即为某个请求采样；请求相关的温度、Top-K、Top-P 由调度器调用 `Sampler.prepare()` 时准备。

接下来引擎再次调用 `_sync_get_memory()`，取最小空闲显存用于日志。这个日志出现在 CUDA Graph 捕获之前，因此还不是图捕获完成后的最终显存状态。

#### 3.2.8. 设置占位请求并捕获 CUDA Graph

引擎创建一个输入 token 为 0、`uid = -1` 的 `dummy_req`，让它使用额外的表行，并将这一整行填为 `num_tokens`。

正常缓存槽位从 0 到 `num_tokens - 1`，所以 `num_tokens` 指向额外 dummy 页的起始位置。占位请求的采样参数与缓存句柄设为 `None`，它用于图捕获和填充，不按真实请求执行采样结果处理。

随后创建 `GraphRunner`，传入执行流、模型、Attention 后端、图批次档位、对齐后的最大长度、词表大小与占位请求。它在构造过程中分配固定缓冲区，并对启用的 Decode 批次档位预热和捕获模型前向。

这个阶段必须放在模型、权重、缓存和后端准备完成之后，因为捕获过程确实会执行模型。没有启用图档位时，`GraphRunner` 跳过捕获，运行时使用普通前向。

这里传给 `GraphRunner` 的 `free_memory` 是加载模型前记录的 `init_free_memory`，用于选择默认图档位；它不是刚打印的剩余显存，也不是为图捕获单独预留的内存额度。

### 3.3. _init_communication：选择 TP 通信路径

该方法返回供 CPU 数据通信使用的 `tp_cpu_group`，主要分成两个分支。

| 条件                                  | PyTorch 默认进程组 | GPU 张量通信                                             |
| ------------------------------------- | ------------------ | -------------------------------------------------------- |
| TP 规模为 1，或启用`use_pynccl`     | Gloo               | 多 rank 时注册项目的 PyNCCL 实现；单 rank 时无需跨卡通信 |
| TP 规模大于 1，且未启用`use_pynccl` | NCCL               | 使用 PyTorch 分布式通信，并额外创建 Gloo 组处理 CPU 数据 |

两个分支都使用配置中的 rank、world size、连接地址和超时时间。单 rank 也会初始化进程组，便于后续显存归约等代码使用统一调用方式；`enable_pynccl_distributed()` 在单 rank 时直接返回。

PyNCCL 分支还计算：

```python
max_bytes = config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
```

这对应“最大前向 token 数 × 隐藏维度 × 每个元素字节数”，作为初始化通信器时传入的最大数据规模参数。它不是模型全部权重的大小，也不是 KV Cache 容量。

例如长度为 4096、隐藏维度为 4096、每个元素为 2 字节时，该值为 32 MiB。模型层通过项目的通信接口调用 All-Reduce、All-Gather，实际使用哪条路径由这里的初始化决定。

### 3.4. _load_weight_state_dict：准备真实设备权重

```python
def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
    if config.use_dummy_weight:
        return {
            k: torch.randn_like(v, device=self.device)
            for k, v in self.model.state_dict().items()
        }
    else:
        return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}
```

启用 `use_dummy_weight` 时，方法按照模型已有张量的形状和类型，在实际设备上生成随机权重。它适合测试执行流程或性能，但这些随机值不具备预训练模型的语义能力。

真实权重路径调用 `models/weight.py` 中的 `load_weight()`。该加载器负责读取 checkpoint、按 TP rank 切分张量，以及将分开的 Q/K/V、gate/up 等权重组合成运行时需要的形式。这里再统一转换成 `self.dtype`，收集为按参数名索引的字典。

方法只负责准备字典，真正放入模型属性的是构造函数随后调用的 `self.model.load_state_dict()`。多卡切分也主要由加载器与模型层处理，不是在这个方法中手动计算切片。

### 3.5. _sync_get_memory：汇总各 rank 的空闲显存

#### 3.5.1. 测量前的设备处理

方法依次同步当前设备、释放分配器中未使用的缓存块、重置峰值显存统计，再通过 `get_free_memory()` 读取当前空闲显存。项目中的 `get_free_memory()` 实际返回 `torch.cuda.mem_get_info(device)[0]`。

重置统计不会释放仍在使用的模型或 KV 张量；这里读取的也是当前空闲值，不是峰值用量。该方法用于初始化过程中的容量测量，并不在每次前向时执行。

#### 3.5.2. 用一次 MIN 归约同时得到最小值与最大值

```python
free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
torch.distributed.all_reduce(
    free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
)
min_free_memory = int(free_mem_tensor[0].item())
max_free_memory = -int(free_mem_tensor[1].item())
```

第一项直接得到最小空闲值，第二项通过“先取负数，再求最小值，最后取负”得到最大空闲值。

例如两个 rank 分别空闲 40 GiB 和 39 GiB，归约后的逻辑结果是 `[39, -40]` GiB，返回 `(39, 40)` GiB；真实代码以字节为单位存储整数。

如果最大值与最小值相差超过 2 GiB，方法记录错误并抛出异常，阻止显存状态明显不均衡的 TP 实例继续初始化。所有 rank 需要参与这次归约，因此它不是可以只在某一个 rank 随意调用的本地查询。

### 3.6. _determine_num_pages：将显存预算换算为缓存页

#### 3.6.1. 计算单页成本

```python
cache_per_page = (
    2
    * config.model_config.head_dim
    * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
    * config.page_size
    * self.dtype.itemsize
    * config.model_config.num_layers
)
```

系数 2 对应 Key 和 Value。其余因子依次为 head 维度、当前 rank 的 KV head 数、每页 token 数、每个元素的字节数以及模型层数。因此这里计算的是一个 rank 上，一页 token 在所有层的 KV 总成本。

`allow_replicate=True` 处理 KV head 数小于 TP 规模的情况：满足整除条件时，每个 rank 使用一个 KV head，并在部分 rank 间复制对应 head，而不是计算出不足一个 head 的缓存。

例如 head 维度为 128、每 rank 有 2 个 KV head、每页 16 个 token、元素占 2 字节、模型有 32 层，则每页成本为：

```text
2 × 128 × 2 × 16 × 2 × 32 = 524288 字节 = 512 KiB
```

#### 3.6.2. 计算可分配页数

未指定 `num_page_override` 时，代码使用以下预算：

```python
model_memory = old_free_memory - new_free_memory
available_memory = int(config.memory_ratio * old_free_memory) - model_memory
num_pages = available_memory // cache_per_page
```

`old_free_memory` 是加载模型前的测量值，`new_free_memory` 是加载后的测量值。两者之差用来估算模型加载阶段的显存开销；它是测量差值，不是逐个参数累加得到的精确权重大小。

假设加载前空闲 40 GiB，加载后空闲 30 GiB，`memory_ratio = 0.9`，则模型阶段开销为 10 GiB，缓存预算为 `40 × 0.9 - 10 = 26 GiB`。它不是简单地把加载后剩余的 30 GiB 乘以 0.9。

需要注意，这两个测量值按当前源码都取各 TP rank 的最大空闲值。前面的差异检查只限制不均衡程度，并没有将估算改为最小空闲值；不能把这里解释成按最紧张的 GPU 自动兜底。

#### 3.6.3. 显式覆盖与预算边界

设置 `num_page_override` 后，直接采用指定页数，跳过上述自动预算公式。无论哪条路径，代码都会检查 `num_pages > 1`，然后记录正常缓存的 token 容量与 K、V 大小。

这不是对后续全部显存分配的保证：额外 dummy 页、页表、后端工作区与 CUDA Graph 都还需要显存。`memory_ratio` 留出的空间供后续开销使用，而手动指定过大的页数仍可能在实际分配时失败。

### 3.7. forward_batch：执行一轮推理

#### 3.7.1. 输入与前置条件

方法接收 `batch: Batch` 和 `args: BatchSamplingArgs`。`batch` 提供本轮模型输入与元数据；`args` 提供采样器所需的批量参数，两者应对应相同的真实请求顺序。

```python
assert torch.cuda.current_stream() == self.stream
```

这个断言检查调用者是否已经切到引擎流。方法不会在内部自动替换调用者的当前流，也不会重新完成调度器侧的批次准备。

#### 3.7.2. 在当前批次上下文中执行模型

```python
with self.ctx.forward_batch(batch):
    if self.graph_runner.can_use_cuda_graph(batch):
        logits = self.graph_runner.replay(batch)
    else:
        logits = self.model.forward()
```

进入上下文后，模型层可以通过 `get_global_ctx().batch` 取得本轮输入，因此普通前向不必显式传入 `input_ids`。

图执行条件由 `GraphRunner` 判断：当前批次必须为 Decode，且真实请求数不超过已准备的最大图档位。调度器此前已完成填充，回放时把本轮张量复制到固定缓冲区、准备后端回放元数据，再执行对应的图。

Prefill 或不满足图条件的批次走普通模型前向。这里捕获和回放的主要是模型计算，后续请求状态更新、采样和 CPU 结果复制仍在这段 `with` 之后执行。

#### 3.7.3. 推进请求状态并采样

```python
for req in batch.reqs:
    req.complete_one()

next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
```

前向之后，对真实请求调用 `complete_one()`：将本轮输入计入已缓存长度，并为下一 token 推进逻辑长度。这个操作更新 Python 侧的状态，不等于此时 CPU 已经收到生成结果。

`logits[:batch.size]` 只保留真实请求对应的输出，避免对图填充部分生成业务结果。采样器根据 `args` 选择贪心或随机采样路径，返回每个真实请求的下一 token，并统一转换为 `int32`。

例如真实请求数为 3、图执行填充到 4 时，最终返回 3 个 token。对于分块 Prefill 的中间块，调度器后续会跳过其返回 token 的处理，继续推进输入块。

#### 3.7.4. 返回 CPU 副本与完成事件

```python
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)
return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

引擎使用非阻塞选项发起到 CPU 的复制，并在同一条流上记录完成事件。事件用于标记之前排入该流的工作完成，调用者需要在读取 CPU 结果前等待它，而不能仅凭 Python 方法已经返回就判断复制完成。

`ForwardOutput` 是这个文件定义的 `NamedTuple`，同时提供三项结果：

| 返回项              | 后续用途                                              |
| ------------------- | ----------------------------------------------------- |
| `next_tokens_gpu` | 调度器写回 GPU`token_pool`，供下一轮输入使用        |
| `next_tokens_cpu` | 调度器追加到请求的 CPU 序列，判断 EOS 并组织返回消息  |
| `copy_done_event` | 调度器在处理 CPU 结果前调用`synchronize()` 等待完成 |

分别保留设备结果与主机结果，使 GPU 上的后续输入准备不必依赖 CPU 序列追加。调度器仍需正确安排流间依赖，并在消费 CPU 数据前等待事件；这也是重叠调度能够拆分前向提交与结果处理的基础。

### 3.8. shutdown：按依赖顺序清理资源

```python
def shutdown(self) -> None:
    self.graph_runner.destroy_cuda_graphs()
    torch.distributed.destroy_process_group()
    destroy_distributed()
```

先销毁 CUDA Graph，再销毁 PyTorch 进程组，最后清理项目注册的分布式通信实现。`graph.py` 明确要求在释放 NCCL 资源前销毁图，以避免退出时挂起，因此这里的顺序具有实际意义。

调度器调用引擎关闭方法之前，还会同步设备并同步各 rank。`Engine.shutdown()` 本身没有等待所有业务请求自然完成的循环，也没有逐个删除模型、KV Cache 或重置全局 `Context`。它负责这里列出的图与通信清理，不是完整的引擎重建接口。

### 3.9. 文件级辅助函数

#### 3.9.1. _align_up_32：对齐页表列数

```python
def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32
```

先加 31 再整除 32，最后乘回 32，得到不小于输入的最小 32 倍数。例如 `1000 → 1024`，`1024 → 1024`。构造函数用它准备页表列数，并将对齐后的长度传给 `GraphRunner`。

#### 3.9.2. _adjust_config：落实自动后端配置

`EngineConfig` 是 `frozen=True` 的 dataclass，通常不允许直接修改属性。该函数内部通过 `object.__setattr__()` 实现受控覆盖，因此调用后改变的是原配置对象，而不是创建新的配置副本。

当前源码的自动选择顺序为：

| 条件                                                   | 配置结果                         |
| ------------------------------------------------------ | -------------------------------- |
| Attention 为`auto`，且 `is_sm100_supported()` 为真 | `trtllm`                       |
| 否则，`is_sm90_supported()` 为真                     | `fa,fi`                        |
| 其余自动选择情况                                       | `fi`                           |
| Attention 名称包含`trtllm`，页大小不在 16、32、64 中 | 将`page_size` 改为 64          |
| 模型为 MoE，且 MoE 后端为`auto`                      | 将`moe_backend` 改为 `fused` |

其中 `fa,fi` 会由 Attention 工厂解析为混合后端：Prefill 使用 FlashAttention，Decode 使用 FlashInfer。它不是按顺序尝试两个库是否可用的回退列表。

这个函数解决的是本地实现的默认选择与兼容配置，不负责穷尽所有运行条件的验证。显式选择后端时不会执行 `auto` 分支，但包含 `trtllm` 时仍会应用页大小调整。

### 3.10. 从初始化连接到连续生成

以一个普通请求为例，引擎启动时先建立通信、加载权重、创建缓存与后端，最后准备 CUDA Graph。请求到来后，调度器组织 Prefill 批次并准备输入，引擎执行前向、生成第一个 token，再将设备结果和 CPU 结果的完成事件交回调度器。

如果请求仍可继续生成，调度器会把新 token 作为下一轮 Decode 输入。之后反复执行同一个 `forward_batch()`，直到调度器根据长度或 EOS 判定结束。Prefill 与 Decode 共用引擎入口，区别体现在批次的输入范围、Attention 元数据以及是否使用图回放。

本节依据本地 `python/minisgl/engine/engine.py`，并结合 `engine/config.py`、`engine/graph.py`、`engine/sample.py`、`distributed/impl.py`、`models/weight.py` 和 `layers/base.py` 核对直接调用关系。后续继续阅读调度器时，可以重点跟踪它如何准备 `Batch`、切换执行流，以及消费 `ForwardOutput`。
