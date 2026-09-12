---
layout: ../../../layouts/PostLayout.astro
category: inference
title: Mini-SGLang 解析（4）
description: 解析 Mini-SGLang 的 KV Cache 存储、分页分配与 Radix 前缀复用
date: 2026-9-12
---
# Mini-SGLang 解析（4）

## 1. 概述

在 [Mini-SGLang 解析（1）](/my-blog/posts/inference/minisgl_1/) 中，我们介绍了 `Req`、`Batch`、`Context` 和 `Engine`。其中，`cached_len` 记录请求已经计算好的 KV 长度，`page_table` 记录 token 对应的缓存位置，`batch.out_loc` 告诉模型将本轮的 K、V 写到哪里。

这篇博客继续沿着这条调用链，解析 KV Cache 如何存储、分配、共享和回收。正文依据本地 Mini-SGLang 源码，版本为 `d0c2ecefca4ade346fb11d0d4ef1191ae73efdbc`；下面的文件路径均相对于该项目根目录。

| 文件 | 主要职责 |
| --- | --- |
| `python/minisgl/kvcache/base.py` | 定义物理缓存池、前缀缓存和句柄接口 |
| `python/minisgl/kvcache/mha_pool.py` | 创建 GPU 上的 K、V 存储并提供写入入口 |
| `python/minisgl/kvcache/naive_cache.py` | 不保留跨请求前缀的实现 |
| `python/minisgl/kvcache/radix_cache.py` | 用 Radix Tree 管理共享前缀、引用计数和淘汰 |
| `python/minisgl/kvcache/__init__.py` | 创建缓存池并注册前缀缓存实现 |
| `python/minisgl/scheduler/cache.py` | 分配物理页、维护页表、连接请求与前缀缓存 |

可以先把数据流画成下面的关系：

```text
请求的 token ID
    │ 匹配相同前缀
    ▼
PrefixCache ──句柄中的物理位置──► 请求 page_table
                                      │
                         本轮待计算位置 │ 得到 out_loc
                                      ▼
模型产生 K、V ── store_kv ──► GPU KV Cache Pool
                                      │
                           Attention 按页表读取
```

前缀树保存的是 token 序列与物理位置的对应关系；真正占用大量显存的 K、V 张量由缓存池保存。调度器的 `CacheManager` 则决定哪些页可以分配给请求。

## 2. KV Cache 缓存的是什么

### 2.1. 为什么 Decode 可以复用历史 K、V

在因果 Attention 中，某个位置只能关注它自己和之前的位置。已有前缀不变时，历史位置的 K、V 可以保留，新一轮只计算新输入 token 对应的 Q、K、V。

以单个 Attention head 为例，本轮位置为 $t$ 时可以写为：

$$
O_t = \operatorname{softmax}\left(\frac{Q_t K_{\leq t}^{\mathsf T}}{\sqrt{d}}\right)V_{\leq t}
$$

缓存使模型不必在每一轮重新执行整个历史前缀的前向计算，但本轮 Attention 仍然需要读取历史 K、V。因此，Decode 的历史读取量仍然会随上下文长度增长。

Mini-SGLang 的 `layers/attention.py` 先拆分 Q、K、V，并处理位置编码，再交给 Attention 后端；后端调用缓存池写入 K、V，然后执行 Attention。历史 Q 不放入这个缓存池，因为后续位置使用的是后续位置自己的 Query。

### 2.2. 请求内复用与跨请求复用

请求内复用是连续生成的基础：一次 Prefill 建立输入的 KV，后续 Decode 不断追加。

跨请求复用则要求新请求命中已保存的 token 前缀。例如两个请求都以同一段系统提示开头，第二个请求可以直接使用第一段前缀的 KV，只计算剩余部分。

这里比较的是 token ID 的连续前缀，不是语义相似度，也不是任意相同子串。后面的 token 即使相同，只要前面的上下文不同，就不能据此复用 KV。

### 2.3. 三种容易混淆的长度

| 数值 | 含义 |
| --- | --- |
| `req.cached_len` | 当前请求已有有效 KV 的 token 数 |
| `req.cache_handle.cached_len` | 当前句柄覆盖的共享前缀长度 |
| 已分配页数 × `page_size` | 已分配容量，可能包含尚未使用的尾部槽位 |

假设页大小为 4，请求计算好了 6 个 token，其中前 4 个已经放入前缀缓存。那么这三个数可以分别为 6、4、8。有效数据、共享范围和物理容量描述的是不同状态。

## 3. base.py：缓存接口与返回值

### 3.1. BaseKVCachePool

```python
class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...
```

`k_cache(index)` 和 `v_cache(index)` 返回指定模型层的缓存，参数 `index` 是层号。`store_kv()` 将本轮新产生的 K、V 写入该层的指定位置。

此外，接口还暴露 `device`、`dtype` 和 `num_layers` 三个属性，供后端获取存储信息。这个类没有分配请求页、匹配 token 或淘汰前缀的方法；这些职责由另外两层承担。

### 3.2. BaseCacheHandle

```python
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...
```

句柄提供两个信息：匹配了多少个 token，以及这些 token 的 KV 位于哪些物理槽位。返回的是位置索引，不是 K、V 张量。

`frozen=True` 限制句柄字段被重新赋值，但不代表其引用的 Radix 节点不能变化。树发生分裂时，节点关系仍然会更新。

接口要求使用匹配结果前锁定句柄。匹配本身不会保护物理页，如果在使用位置索引之前发生淘汰，这些位置就可能被其他请求重新使用。

### 3.3. SizeInfo、InsertResult 与 MatchResult

```python
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

class InsertResult(NamedTuple):
    cached_len: int
    handle: BaseCacheHandle

class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
```

`SizeInfo` 的单位是 token 槽位数。`evictable_size` 表示已经保存、但允许淘汰的前缀；`protected_size` 表示被引用保护的前缀；两者都不包含空闲页。

`InsertResult.cached_len` 特别容易误读：它表示插入之前，树中已经存在的匹配长度。插入后句柄的覆盖长度则在 `result.handle.cached_len` 中。前者用于识别重复分配的物理页，后者用于划分保留前缀与尾部。

`MatchResult` 当前只有 `cuda_handle`。源码中主机分层缓存仍是 TODO，不能把这个结构解释成已经支持 CPU/GPU 两级 KV 迁移。

### 3.4. BasePrefixCache

| 接口 | 作用 |
| --- | --- |
| `match_prefix(input_ids)` | 查找可复用前缀并返回句柄 |
| `insert_prefix(input_ids, indices)` | 登记 token 与已有物理位置的关联 |
| `lock_handle(handle, unlock=False)` | 增减句柄路径上的引用保护 |
| `evict(size)` | 移除可淘汰前缀并返回物理位置 |
| `size_info` | 返回可淘汰和受保护的槽位数量 |
| `reset()` | 约定重置接口，具体实现未必完成 |
| `check_integrity()` | 约定一致性检查接口，具体实现未必执行检查 |

`insert_prefix()` 不负责计算 KV；调用时，相应 KV 应当已经由模型写好。`evict()` 返回槽位后，由调度器把对应页加入空闲列表，它本身不缩小 GPU 缓存张量。

## 4. mha_pool.py：物理缓存池

### 4.1. MHAKVCache 的存储布局

构造函数的核心代码为：

```python
tp_info = get_tp_info()
local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
self._k_buffer = self._kv_buffer[0]
self._v_buffer = self._kv_buffer[1]
self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
```

六个维度依次表示 K/V、模型层、物理页、页内 token、当前 TP rank 的 KV head，以及 head 内部维度。

`_k_buffer` 和 `_v_buffer` 是原张量的切片视图，不会再次各分配一份完整缓存。`torch.empty()` 也不会将数据初始化为有效 KV；只有已经写入且由有效序列长度覆盖的位置才能用于 Attention。

虽然类名中使用 MHA，但布局以 `num_kv_heads` 为依据，也能表示 Q head 与 KV head 数量不同的 GQA 配置。工厂中的 MLA 支持仍是 TODO。

### 4.2. TP 下的 head 数量与显存成本

通常情况下，当前 rank 的 KV head 数等于总 KV head 数除以 TP 规模。若 KV head 数少于 TP 规模，`allow_replicate=True` 在满足对应整除条件时允许复制，每个 rank 至少保有一个 KV head。

设层数为 $L$，页数为 $N$，页大小为 $P$，当前 rank 的 KV head 数为 $H$，head 维度为 $D$，每个元素占 $B$ 字节，则单 rank 的缓存池大小为：

$$
M = 2LNP HDB
$$

例如 `L=32`、`P=16`、`H=2`、`D=128`、`B=2`，每页在全部层上的成本为：

```text
2 × 32 × 16 × 2 × 128 × 2 = 524288 字节 = 512 KiB
```

第一篇中的 `Engine._determine_num_pages()` 正是用这个单页成本将显存预算换算成页数。引擎实际创建池时还会额外增加一个 dummy 页，供 CUDA Graph 占位请求使用；正常请求的空闲页列表不包含它。

### 4.3. 页号与 token 槽位

`k_cache(layer_id)` 返回形状为 `[num_pages, page_size, local_kv_heads, head_dim]` 的张量。写入时则把前两个维度展平成一个槽位维度。

槽位编号满足：

$$
\text{slot} = \text{page\_id}\times P + \text{offset}
$$

例如页大小为 4，槽位 13 表示物理页 3 的页内位置 1。读取视图与写入视图引用同一块存储，只是维度组织不同。

### 4.4. store_kv：按 out_loc 写入

```python
store_cache(
    k_cache=self._k_buffer[layer_id].view(self._storage_shape),
    v_cache=self._v_buffer[layer_id].view(self._storage_shape),
    indices=out_loc,
    k=k,
    v=v,
)
```

用示意代码表达它的语义就是：

```python
# 示意：实际调用的是项目的 CUDA 写入内核
for i in range(len(out_loc)):
    k_cache[out_loc[i]] = k[i]
    v_cache[out_loc[i]] = v[i]
```

例如 `out_loc=[8, 9, 20]`，本轮三个 token 的 KV 分别写入这三个槽位，不要求物理位置连续。不同模型层使用同一组槽位编号，但写入各自层的存储区域。

`kernel/store.py` 进一步展平 head 维度，按每个 token 的存储字节数选择并缓存 JIT 编译模块，再调用 CUDA 内核。这里执行的是数据写入；空闲页管理已经在调度器准备批次时完成。

## 5. naive_cache.py 与缓存工厂

### 5.1. NaivePrefixCache

Naive 实现始终返回 `cached_len=0` 的句柄，匹配位置是空的 GPU `int32` 张量。它的插入结果同样是零长度句柄，`size_info` 的两个数量均为 0。

这表示不保留跨请求前缀，并不表示禁用请求内部的 KV Cache。请求仍然通过 `MHAKVCache` 保存历史 K、V，Decode 仍然复用历史计算；只是请求结束后，对应页全部归还空闲列表。

`lock_handle()`、`reset()` 和 `check_integrity()` 都是空操作。`evict(0)` 返回空张量，非零淘汰则抛出 `NotImplementedError`，因为没有保存可淘汰前缀。

### 5.2. __init__.py 中的工厂

`create_kvcache_pool()` 当前直接创建 `MHAKVCache`，而 `create_prefix_cache(device, type)` 通过注册表选择 `naive` 或 `radix`。

因此，物理存储形式与前缀复用策略是两个独立选择。将前缀策略改为 `naive`，不会将底层 GPU 缓存池替换成另一种存储结构。

## 6. radix_cache.py：共享前缀树

### 6.1. RadixTreeNode 的字段

Radix Tree 将连续的一段 token 压缩到一个节点中，避免每个 token 都创建一个 Python 节点。

| 字段 | 含义 |
| --- | --- |
| `_key` | 该节点这一段 token ID |
| `_value` | 该段 token 对应的 GPU KV 槽位索引 |
| `_length` | 该节点的 token 数，等于 key 和 value 的长度 |
| `_parent`、`children` | 父节点和子节点映射 |
| `ref_count` | 保护该节点的引用数量 |
| `timestamp` | 最近访问的时间，用于淘汰排序 |
| `uuid` | 通过类级计数器生成的节点编号 |
| `key_fn` | 从 token 片段提取子节点查找键 |

`set_key_value()` 要求 key 与 value 等长；`set_parent()` 既记录父节点，也把自身放入父节点的 `children`。根节点没有实际缓存片段，且 `ref_count=1`，始终受保护。

### 6.2. _get_key_fn：按第一页定位分支

```python
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
```

页大小为 1 时，使用第一个 token 作为字典键；页大小大于 1 时，使用第一页 token 的元组。

这与页级共享保持一致：如果第一页都不相同，这条分支就不具备可共享的完整页。找到候选子节点后，还要继续比较整段 key，不能把字典键命中当成整个节点命中。

### 6.3. _tree_walk：查找与页对齐

遍历从根节点开始，使用尚未匹配的输入定位子节点，再调用 `get_match_len()` 得到最长公共前缀长度。后者委托给 `fast_compare_key()`。

随后代码向下对齐匹配长度：

```python
match_len = node.get_match_len(input_ids[prefix_len:])
match_len = align_down(match_len, self.page_size)
prefix_len += match_len
```

例如页大小为 4，两个片段前 7 个 token 相同，实际复用长度只能是 4。余下 3 个相同 token 仍需重新计算，因为该实现只将完整页纳入共享前缀。

如果当前节点整段匹配，则继续向下；没有对应子节点时返回已有匹配；只匹配当前节点的一部分时，则分裂节点，并返回公共前缀节点。

需要注意接口注释与实现的区别：`BasePrefixCache.match_prefix()` 的注释称匹配不修改缓存，但 Radix 实现会更新时间戳，也可能分裂树节点。更准确地说，匹配不新增 KV 数据、不增加缓存总槽位数量，但可能改变树的结构和元数据。

### 6.4. split_at：保留公共前缀

假设页大小为 2，已有节点表示 `[A, B, C, D]`，新的输入为 `[A, B, X, Y]`。分裂后的结构是：

```text
分裂前：
root ── [A B C D]

分裂并插入新后缀后：
root ── [A B]
          ├── [C D]
          └── [X Y]
```

`split_at(2)` 创建新的 `[A, B]` 节点，把原节点修改为 `[C, D]`，再将原节点挂到新节点下面。已有请求如果持有原节点，仍然可以沿父链取得完整的 `[A, B, C, D]`。

新公共节点继承原节点的引用计数和时间戳；原节点的 key、value 则使用后半段切片。拆分不复制实际 K、V，也不改变缓存总长度，因此不需要改变 `evictable_size + protected_size`。

### 6.5. RadixCacheHandle：沿父链还原索引

`get_matched_indices()` 从句柄节点向根遍历，收集每个节点的 value，然后反转并拼接：

```python
while not node.is_root():
    value_list.append(node.value)
    node = node.parent
value_list.reverse()
return torch.cat(value_list)
```

假设公共节点保存 `[8, 9]`，后缀节点保存 `[20, 21]`，返回结果就是 `[8, 9, 20, 21]`。拼接的是小得多的索引张量，没有复制各层 K、V。

当前实现对根句柄直接调用这个方法会遇到空列表拼接问题。实际 Prefill 调用处先判断 `cached_len > 0`，只在有命中时取索引，因此零命中路径不会执行这次拼接。

### 6.6. insert_prefix：登记新的完整页

```python
insert_len = align_down(len(input_ids), self.page_size)
input_ids, indices = input_ids[:insert_len], indices[:insert_len]
node, prefix_len = self._tree_walk(input_ids)
```

插入先丢掉不足一页的尾部，然后查找已有前缀。若还有未登记部分，创建新节点，保存剩余 token 和 `indices[prefix_len:].clone()`，并增加 `evictable_size`。

这里 `clone()` 复制的是位置索引，使缓存树不依赖请求页表切片后续是否被覆盖。K、V 仍留在原物理槽位，插入不会另存一份完整 KV。

例如输入有 10 个 token，页大小为 4，树中已有前 4 个，则：

```text
insert_len                 = 8
InsertResult.cached_len    = 4
返回 handle.cached_len     = 8
新增登记的长度              = 4
无法登记的尾部长度          = 2
```

新节点初始未加锁，因此先归入可淘汰部分。若请求仍在运行，调度器随后会锁定新句柄。

### 6.7. lock_handle：保护整条前缀路径

锁定从句柄节点一直向根遍历，每经过一个非根节点就增加 `ref_count`。仅在 `0 → 1` 时，把该节点长度从可淘汰量移动到受保护量。

解锁则逐级减一，仅在 `1 → 0` 时将该节点重新标记为可淘汰。计数仍大于 0，说明还有其他请求使用它。

例如请求 A 和 B 共享长度为 4 的节点，两者都锁定后，节点的 `ref_count=2`，但 `protected_size` 只计入 4 个 token。A 结束后计数降为 1，这段前缀仍然不能被淘汰。

这个“锁”是引用保护机制，不是用于 Python 线程互斥的锁。

### 6.8. evict：从最旧的可淘汰叶节点开始

`evict(size)` 首先检查可淘汰容量，然后遍历树，收集引用计数为 0 的叶节点，用 `timestamp` 建立最小堆。

每次弹出最旧叶节点，收集它的物理索引，将它从父节点删除，并减少 `evictable_size`。如果删除后父节点也变成未被引用的叶节点，就把父节点加入堆，继续回收。

这样可以保持前缀依赖：仍有子节点时，不能先把公共父前缀删除。由于按照节点整段回收，请求淘汰 4 个 token，实际可能回收 8 个或更多。

这是基于访问时间的叶节点淘汰策略。它受到树结构和引用保护限制，不是所有 token 都能独立参与的全局 LRU。

当前 `reset()` 仍抛出 `NotImplementedError`，`check_integrity()` 仍为 `pass`。理解接口时应区分设计约定与已经落实的检查。

## 7. scheduler/cache.py：页分配与请求生命周期

### 7.1. free_slots 保存页首槽位

```python
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

当页大小为 4、正常页数为 4 时，初始化结果为 `[0, 4, 8, 12]`。每个元素表示一整页的起始槽位，而不是页号。

`_page_to_token()` 将页首展开成每个 token 的物理位置，例如：

```text
输入页首：[8, 20]
展开槽位：[8, 9, 10, 11, 20, 21, 22, 23]
```

同一请求的逻辑页可以对应不同物理页，不要求相邻。

### 7.2. available_size 与前缀匹配

```python
return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
```

`available_size` 包含空闲容量和可通过淘汰获得的容量，单位是 token 槽位。它不是当前空闲显存，也不包含正在保护的共享前缀。

匹配入口则刻意少取最后一个输入 token：

```python
return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

因为这里只缓存 K、V，没有缓存最后位置的 logits；需要至少执行一个输入 token 的前向来产生下一 token 预测。这也满足第一篇介绍的 `cached_len < device_len`。

如果完整输入有 8 个 token，页大小为 4，即使树中已有完全相同的 8 个 token，查询也只传入前 7 个，实际最多复用前 4 个。

Prefill 接纳请求时先估算容量，再锁定句柄，然后再次检查容量。第二次检查是必要的：此前计入可淘汰空间的匹配前缀，加锁后可能已转入受保护空间。检查通过后，才分配请求表行并复制命中的索引。

### 7.3. allocate_paged：只补足本轮缺少的页

```python
first_page = div_ceil(req.cached_len, self.page_size)
last_page = div_ceil(req.device_len, self.page_size)
```

新增页数是 `last_page - first_page`。这里向上取整已缓存长度，是因为不足一页的有效 KV 已经占用了完整物理页，后续 token 可以继续使用这页剩余槽位。

以页大小 4 为例：

| `cached_len` | `device_len` | 已覆盖页数 | 本轮需要页数 | 新增页数 |
| --- | --- | --- | --- | --- |
| 0 | 6 | 0 | 2 | 2 |
| 6 | 7 | 2 | 2 | 0 |
| 7 | 8 | 2 | 2 | 0 |
| 8 | 9 | 2 | 3 | 1 |

代码先汇总整个批次所需页数，一次性分配并展开槽位，再由 `_write_page_table()` 写入各请求的对应区间。

`_write_page_table()` 在 pinned CPU 内存中构造表行号和逻辑位置，传到 GPU 后执行索引赋值。页表会写入整页映射，即使本轮只用其中一个槽位；有效读取范围仍由请求长度控制。

### 7.4. _allocate 与 _free

如果空闲页足够，`_allocate()` 直接从 `free_slots` 开头取出指定数量。否则先要求前缀缓存淘汰缺少的容量，再将返回索引按 `indices[::page_size]` 转回页首。

`_free()` 同样只抽取每页的第一个槽位，再追加到空闲列表。这依赖调用者提供从页边界开始、按页组织的索引区间；它不是处理任意散乱 token 索引的通用释放函数。

归还最后一个不满页的有效片段时，也能借助它的首槽位归还整页。例如 `[20, 21]` 是页大小 4 的尾页已用部分，提取 `[20]` 就表示释放整页 `[20, 21, 22, 23]`。

这些操作改变分配状态，不会调用 `cudaFree`，也不会将旧 K、V 清零。后续请求会覆盖重用的槽位。

### 7.5. cache_req：插入、去重和尾部处理

核心流程为：

```python
insert_ids = req.input_ids[: req.cached_len]
page_indices = self.page_table[req.table_idx, : req.cached_len]
old_handle = req.cache_handle
cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
self.unlock(old_handle)
self._free(page_indices[old_handle.cached_len : cached_len])
if finished:
    self._free(page_indices[new_handle.cached_len :])
else:
    req.cache_handle = new_handle
    self.lock(new_handle)
```

这里存在四个边界，可分别记为 $O$、$E$、$N$、$C$：旧句柄长度、插入前树中已有的长度、新句柄长度、请求有效 KV 长度。

| 区间 | 含义与处理 |
| --- | --- |
| `[0, O)` | 请求原先就复用的共享前缀，不能按私有页释放 |
| `[O, E)` | 请求自己计算过，但插入时树中已经存在；释放请求的重复页 |
| `[E, N)` | 本次新登记到树中的完整页，保留供共享 |
| `[N, C)` | 不足完整页的尾部；继续运行则保留，结束则归还 |

出现 `[O, E)` 是因为多个请求可能在各自匹配之后计算了相同前缀，随后先后插入。后插入者应当释放重复分配的页，避免同一前缀长期占两份存储。

需要结合源码保留一个实现边界：当前 `cache_req(finished=False)` 更新了句柄，但没有把页表的重复区间改写成新句柄中的规范索引。如果运行中请求确实出现 `E > O`，仅凭这个函数无法保证后续读取安全；还需要核实调度路径是否避免了该情形，或补上页表重映射。这里描述的是源码已有的去重意图与操作，不将未实现的重映射当成已完成行为。

### 7.6. lazy_free_region 与一致性检查

`lazy_free_region()` 临时替换实例上的 `_free`，将多个待释放片段的页首收集起来，退出区域时再统一拼接到 `free_slots`，减少逐请求拼接张量的开销。

这个方法没有自行创建 CUDA 完成事件。“延迟”指的是区域退出时集中更新空闲列表，设备操作的正确顺序仍依赖外层调度器和执行流。

`check_integrity()` 检查：

```text
空闲页数 + 前缀树中保存的页数 == 正常总页数
```

并检查空闲页首是否对齐。调度器在空闲时调用它；运行中请求尚未插入树的私有页没有计入这个等式，因此不应把它理解为任意运行时刻都适用的完整资源审计。Radix 自身的一致性检查当前也是空实现。

## 8. 与 Attention、Prefill 和 Decode 的连接

### 8.1. 页表如何变成读写地址

调度器准备批次时先执行 `allocate_paged()`，再根据请求表行和本轮输入位置取出 `batch.out_loc`。若某请求页表为：

```text
逻辑位置：0  1  2  3   4   5   6   7
物理槽位：8  9 10 11  20  21  22  23
```

本轮只计算位置 4、5 时，写入位置就是 `[20, 21]`。Attention 读取的范围则可以包含位置 0 到 5，因此写入索引与历史读取索引不是同一个范围。

全局 `page_table` 的每一项是 token 槽位编号。FlashAttention 后端每隔 `page_size` 取一个页首，再除以页大小，转换为内核需要的物理页号。上面的映射转换后为 `[2, 5]`。

FlashInfer 路径则把缓存视图展平成页大小为 1 的形式，并使用 token 粒度索引。后端适配改变索引表示方式，不会搬动实际 K、V。

### 8.2. Prefill 后并不是所有新 token 都已有 KV

`Engine.forward_batch()` 在模型前向后调用：

```python
self.cached_len = self.device_len
self.device_len += 1
```

随后采样得到的 token 尚未经过自己的前向，因此它的 KV 要到下一轮才计算。缓存插入必须截止到 `req.cached_len`，不能直接把完整 `input_ids` 都当成有效 KV。

普通 Prefill 的结果处理会调用 `cache_req(finished=False)`，将已完成的完整页登记并保护起来。普通 Decode 不会每轮都调用这个方法；新增 KV 可以暂时保持为请求私有页，直到结束等回收路径再插入。

对于 `ChunkedReq`，调度器跳过普通的采样结果处理；后续块沿用已有表行、句柄和 `cached_len`，继续补齐输入。不能将中间块误认为已经完成并归还资源。

### 8.3. 一个完整例子

假设页大小为 4，没有重叠执行，也没有其他请求竞争，输入为 `[A, B, C, D, E, F]`，最多生成 3 个 token，且不提前遇到 EOS。

1. 初次匹配未命中，`cached_len=0`、`device_len=6`。分配两页，容量为 8 个 token。
2. Prefill 计算 6 个输入的 KV，采样得到 `G`。更新后 `cached_len=6`、`device_len=7`。前 4 个 token 登记到 Radix Tree，后 2 个仍位于请求私有尾页。
3. 第一轮 Decode 输入 `G`，复用尾页空位，计算位置 6 的 KV，采样得到 `H`。更新后长度分别为 7 和 8。
4. 第二轮 Decode 输入 `H`，仍复用尾页，采样得到 `I`。更新后长度分别为 8 和 9，请求用完输出预算。
5. 请求结束时，只有 `[A, B, C, D, E, F, G, H]` 具有 KV。这 8 个 token 可以整页登记到树中；`I` 没有执行后续前向，不属于缓存内容。没有其他引用时，这两页成为可淘汰缓存。

之后一个新请求以 `[A, B, C, D, X, Y]` 开头，就可以复用前 4 个 token 的 KV，只计算 `X、Y`。新请求的页表前半部分指向原物理页，不需要为共享前缀复制所有层的 K、V。

请求结束释放的是表行与独占资源；已登记的共享前缀可以继续占据缓存池，直到空间不足时被淘汰。若使用 Naive 策略，则结束时这两页直接回到空闲列表。

## 9. 源码阅读入口

回看第一篇中的四个字段，可以将它们与本篇实现一一对应：

| 第一篇中的字段 | 本篇中的实现关系 |
| --- | --- |
| `Req.cached_len` | 决定下一轮计算起点、页分配起点和插入有效范围 |
| `Req.cache_handle` | 连接前缀树节点，保护已复用的完整页 |
| `Context.page_table` | 将请求逻辑位置映射到物理 token 槽位 |
| `Batch.out_loc` | 将本轮新 K、V 写入选定的物理槽位 |

本文除 `kvcache/` 与 `scheduler/cache.py` 外，还核对了 `scheduler/prefill.py`、`scheduler/scheduler.py`、`engine/engine.py`、`layers/attention.py`、`attention/fa.py`、`attention/fi.py` 和 `kernel/store.py` 的直接调用关系。后续阅读 Attention 后端时，可以继续跟踪逻辑序列长度、页表转换和因果掩码如何一起限定有效读取范围。
