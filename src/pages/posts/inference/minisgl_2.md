---
layout: ../../../layouts/PostLayout.astro
category: inference
title: Mini-SGLang 解析（2）
description: 解析 Mini-SGLang 的模型结构、基础算子、张量并行与权重加载
date: 2026-9-12
---
# Mini-SGLang 解析（2）

## 1. 概述

在 [Mini-SGLang 解析（1）](/my-blog/posts/inference/minisgl_1/) 中，我们介绍了推理的基本流程：引擎创建模型并加载权重，调度器准备 `Batch`，然后引擎在批次上下文中执行 `self.model.forward()`，取得 logits 并采样。

这篇博客继续分析模型内部发生了什么。从模型配置与注册入口开始，沿着 Embedding、Decoder Layer、Attention、MLP 和 LM Head 的顺序阅读源码，最后解释 checkpoint 如何转换成这些层实际使用的权重。

本文依据本地 Mini-SGLang 源码，版本为 `d0c2ecefca4ade346fb11d0d4ef1191ae73efdbc`。下面的源码路径均相对于该项目根目录，代码块中的“示意”表示为解释数据流而简化的写法。

| 文件 | 主要职责 |
| --- | --- |
| `python/minisgl/models/config.py` | 将 Hugging Face 配置转换为内部模型配置 |
| `python/minisgl/models/register.py`、`__init__.py` | 按架构名称创建模型实例 |
| `python/minisgl/models/llama.py` 等 | 组织模型、Decoder Layer 和输出头 |
| `python/minisgl/models/utils.py` | 组合共享的 Attention、Dense MLP 和 MoE MLP |
| `python/minisgl/layers/` | 实现权重管理、线性层、归一化、位置编码等基础算子 |
| `python/minisgl/models/weight.py` | 读取、切分、合并和堆叠 checkpoint 权重 |

模型前向的主要路径可以表示为：

```text
Context.batch.input_ids
          │
          ▼
VocabParallelEmbedding
          │
          ▼
Decoder Layer × num_layers
  ├─ 残差合并与 RMSNorm
  ├─ QKV 投影 → 可选 QK Norm → RoPE → Attention → O 投影
  ├─ 残差合并与 RMSNorm
  └─ Gated MLP 或 MoE MLP
          │
          ▼
最终残差合并与 RMSNorm
          │
          ▼
ParallelLMHead → logits → 引擎中的 Sampler
```

模型负责从 token 计算 logits，采样和请求停止判断仍由引擎与调度器完成。

## 2. config.py 与 register.py：确定模型结构

### 2.1. ModelConfig 的参数

`ModelConfig` 是 `frozen=True` 的 dataclass，用一组统一字段描述当前实现支持的模型结构。

| 字段 | 含义 |
| --- | --- |
| `num_layers` | Decoder 层数 |
| `num_qo_heads` | Query 与 Attention 输出的 head 数 |
| `num_kv_heads` | Key、Value 的 head 数 |
| `head_dim` | 每个 Attention head 的维度 |
| `hidden_size` | 残差流与模型隐藏状态的维度 |
| `vocab_size` | 词表大小 |
| `intermediate_size` | Dense MLP 的中间维度 |
| `rms_norm_eps` | RMSNorm 分母中的数值稳定项 |
| `rotary_config` | RoPE 的维度、位置范围、频率基数和缩放配置 |
| `hidden_act` | Dense MLP 的激活函数名称 |
| `tie_word_embeddings` | 输入 Embedding 与输出头是否共享权重 |
| `num_experts`、`num_experts_per_tok` | 专家总数和每个 token 选择的专家数 |
| `moe_intermediate_size` | 每个专家的中间维度 |
| `norm_topk_prob` | 是否对选中专家的路由权重重新归一化 |
| `model_type`、`architectures` | 模型类型与架构名称 |

这里记录的是完整模型的 head 数和中间维度。各算子在构造时再结合 TP 规模，计算本 rank 的局部形状。

`hidden_size` 不应无条件等同于 `num_qo_heads × head_dim`。源码允许配置显式给出 `head_dim`，QKV 投影负责从隐藏维度映射到 Attention 维度，O 投影再映射回来。

### 2.2. RotaryConfig

```python
@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None
```

`head_dim` 是完整 head 维度，`rotary_dim` 是参与旋转的位置编码维度。当前 `from_hf()` 将两者设为相同值，底层 `RotaryEmbedding` 也断言它们相等，因此这里实际使用完整 head 的 RoPE。

`max_position` 决定预计算位置表的长度，`base` 决定基础频率，`scaling` 用于选择并配置频率调整方式。

### 2.3. from_hf：统一外部配置

该方法读取 Hugging Face 的 `PretrainedConfig`，将不同模型的字段映射到上述内部字段。例如：

```python
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
```

没有单独的 KV head 数时，默认与 Query head 数相同；没有有效 `head_dim` 时，才通过隐藏维度除以 Query head 数推导。

如果配置包含 `text_config`，则改为读取文本子配置；当子配置缺少有效值时，再从外层补入 `architectures`、`rope_theta` 和 `rope_scaling`。这为带外层包装的文本模型提供入口，但不意味着模型前向已经实现视觉编码或多模态融合。

专家数量优先读取 `num_local_experts`，其次读取 `num_experts`。`is_moe` 则直接判断 `model_type` 是否包含字符串 `"moe"`，没有依据专家数量自动推断。

RoPE 基数先取 `rope_theta`，否则读取 `rope_scaling["rope_theta"]`。因此，这个转换函数面向当前支持的配置格式，不是对任意 Hugging Face 配置都成立的通用转换器。

### 2.4. create_model：按架构名称创建实例

```python
def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)
```

工厂只读取 `architectures` 的第一个名称。`get_model_class()` 查找注册表、延迟导入对应模块，然后调用模型类的构造函数；虽然函数名中有 `class`，实际返回的是模型实例。

| 架构名称 | 实现入口 |
| --- | --- |
| `LlamaForCausalLM` | `models/llama.py` |
| `Qwen2ForCausalLM` | `models/qwen2.py` |
| `Qwen3ForCausalLM` | `models/qwen3.py` |
| `Qwen3MoeForCausalLM` | `models/qwen3_moe.py` |
| `MistralForCausalLM` | `models/mistral.py` |
| `Mistral3ForConditionalGeneration` | 同样映射到 `MistralForCausalLM` |

未注册的架构会抛出 `ValueError`。注册表决定实际可创建哪些模型，不能仅凭配置能够解析就判断某个架构已经受支持。

## 3. layers/base.py：轻量算子与权重管理

### 3.1. BaseLLMModel 与 BaseOP

`BaseLLMModel` 继承 `ABC` 和 `BaseOP`，约定无显式输入参数的 `forward()` 返回张量。真正的批次输入通过 `get_global_ctx().batch` 获取。

`BaseOP` 则定义算子的 `forward()`、`state_dict()` 和 `load_state_dict()`。它不是 `torch.nn.Module` 的子类：权重是普通 Tensor，对象调用通常显式写成 `.forward()`，状态收集也使用项目自己的递归逻辑。

这与第一篇中的 meta 初始化相配合：先构造张量形状，再把真实设备权重替换到相应属性中。

### 3.2. state_dict：从公开属性收集权重

核心规则是：遍历 `self.__dict__`，跳过以下划线开头的属性；遇到 Tensor 就记录，遇到 `BaseOP` 就递归。

例如：

```text
LlamaForCausalLM
  └─ model
      └─ layers
          └─ 0
              └─ self_attn
                  └─ qkv_proj
                      └─ weight

得到键：model.layers.0.self_attn.qkv_proj.weight
```

整数、函数等普通属性不进入状态字典。以下划线开头的通信对象、临时缓存等同样不参与权重保存。

因此，属性命名本身构成权重命名协议的一部分。调整对象结构或属性名称时，必须同步考虑 checkpoint 的名称转换。

### 3.3. load_state_dict：替换张量，而不是复制到 meta 存储

```python
item = state_dict.pop(_concat_prefix(prefix, name))
assert isinstance(item, torch.Tensor)
assert param.shape == item.shape and param.dtype == item.dtype
setattr(self, name, item)
```

加载器逐项取出权重，核对形状和 dtype，再将属性指向真实张量。它不对 meta 张量执行 `copy_()`，因此能够将 meta 占位权重替换成 GPU 权重。

该方法会消费传入字典：缺失键在 `pop()` 处报错；最外层递归完成后如果仍有键，则抛出 `Unexpected keys`。这也意味着调用后不能指望原字典仍然保留全部权重。

这里没有自动进行 dtype 转换。第一篇中的引擎先将载入权重转换成 `config.dtype`，再调用此方法，确保与初始化的占位张量类型一致。

### 3.4. OPList 与 StateLessOP

普通 Python 列表不会被 `BaseOP.state_dict()` 自动递归，因此模型用 `OPList` 包装 Decoder 列表。它显式按 `0、1、2……` 为元素生成路径，形成 `model.layers.0...` 这样的键。

`StateLessOP` 覆盖状态读写，不收集或加载内部字段。这里的“无状态”指不参与 checkpoint 状态管理，并不是对象完全不能持有运行时缓存或其他算子的引用。

例如 `AttentionLayer` 持有 RoPE 与 Q/K Norm 引用，但自身不重复保存它们的权重。可学习的 Q/K Norm 已作为 `RopeAttn.q_norm`、`RopeAttn.k_norm` 登记，避免同一组参数出现两套加载路径。

## 4. llama.py：完整模型的前向骨架

### 4.1. LlamaForCausalLM

最外层有两个主要成员：`model` 是产生隐藏状态的 `LlamaModel`，`lm_head` 把隐藏状态映射到词表分数。

```python
def forward(self) -> torch.Tensor:
    output = self.model.forward(get_global_ctx().batch.input_ids)
    logits = self.lm_head.forward(output)
    return logits
```

`Context` 必须已经由引擎设置好当前批次。这个方法不创建 `Batch`、不分配 KV 页，也不读取 CPU 请求列表来逐条生成。

### 4.2. LlamaModel

```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    x = self.embed_tokens.forward(input_ids)
    residual: torch.Tensor | None = None
    for layer in self.layers.op_list:
        x, residual = layer.forward(x, residual)
    return self.norm.forward(x, residual)[0]
```

设本轮展平后的输入 token 总数为 $T$，隐藏维度为 $H$。Embedding 将 `[T]` 的 token ID 转成 `[T, H]` 的隐藏状态，之后各层围绕这个 token 维度执行。

Prefill 时，$T$ 是各请求本轮待计算 token 数之和，不一定包含已经命中缓存的前缀。Decode 时通常每个请求提供一个 token；使用 CUDA Graph 时还可能包含占位 token。

不同请求可以拼接在同一个张量中。请求边界、序列长度和历史 KV 位置由 Attention 元数据描述，而不是由普通线性层分别处理。

### 4.3. LlamaDecoderLayer

```python
def forward(self, x, residual=None):
    x, residual = self.input_layernorm.forward(x, residual)
    x = self.self_attn.forward(x)
    x, residual = self.post_attention_layernorm.forward(x, residual)
    x = self.mlp.forward(x)
    return x, residual
```

每层持有两个 RMSNorm、一个 Attention 和一个 MLP。乍看这段代码，似乎没有常见的 `x = x + attention_output` 和 `x = x + mlp_output`，实际上残差相加已经放进 `RMSNormFused` 中。

`_layer_id` 用于 NVTX 标注，方便在性能分析工具中区分各层。它以下划线开头，不属于 checkpoint 参数。

### 4.4. RMSNorm 的计算

对某个 token 的隐藏向量 $x$，RMSNorm 可以表示为：

$$
\operatorname{RMSNorm}(x)_i
= w_i\frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2+\epsilon}}
$$

它使用均方根进行缩放，不执行减均值。`RMSNorm` 保存一维可学习权重 `weight` 和 `eps`，将计算委托给 FlashInfer 的 `rmsnorm`；`forward_inplace()` 通过 `out=x` 将归一化结果写回原张量。

普通层归一化的权重长度是 `hidden_size`；Q/K Norm 的权重长度则是 `head_dim`，两者归一化的最后一维不同。

### 4.5. RMSNormFused：延迟合并残差

```python
if residual is None:
    return self.rmsnorm(x, self.weight, self.eps), x
self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
return x, residual
```

第一层输入尚无独立 residual，先返回归一化结果，同时把原始 Embedding 输出保留为残差。后续路径将残差加法与 RMSNorm 合并：更新 residual 为二者之和，再把归一化结果写入 x。

以一层的数学计算为例：

$$
a = \operatorname{Attention}(\operatorname{RMSNorm}_1(h))
$$

$$
r = h+a,\qquad m=\operatorname{MLP}(\operatorname{RMSNorm}_2(r))
$$

这一层返回 `(m, r)`，没有立即计算最终的 `r + m`。下一层的 `input_layernorm` 才把两者相加，再归一化；如果已经是最后一层，就由模型末尾的 `self.norm` 完成合并。

因此，层间传递的 `x` 可以只是上一层 MLP 的输出，`residual` 才保存此前的累计残差。阅读中间张量时，应把这两个返回值一起理解。

## 5. embedding.py：输入与输出的词表并行

### 5.1. VocabParallelEmbedding 的参数与分片

构造函数接收 `num_embeddings` 与 `embedding_dim`，对应词表大小 $V$ 和隐藏维度 $H$。设 TP 规模为 $p$，每 rank 的分配行数为：

$$
V_{\mathrm{local}}=\left\lceil\frac{V}{p}\right\rceil
$$

rank $r$ 负责从 `r × V_local` 开始的一段词表，`vocab_range` 保存其全局起点和实际有效长度。权重形状为 `[V_local, H]`。

以 `V=8`、`p=2` 为例，rank 0 保存 token 0 到 3 的向量，rank 1 保存 token 4 到 7 的向量。

### 5.2. forward：本地查表，再求和

```python
y = indexing(
    weights=self.weight,
    indices=x,
    vocab_range=self.vocab_range if self.tp_size > 1 else None,
)
return self._comm.all_reduce(y) if self.tp_size > 1 else y
```

项目的 `indexing` 内核对落在本 rank 词表范围内的 token 执行本地查表，对其他 token 输出零向量。随后通过 All-Reduce 求和，使每个 rank 都得到完整的 `[T, H]` 隐藏状态。

例如输入 token 为 5，rank 0 输出零，rank 1 输出 token 5 的 Embedding；求和后两张卡都拿到这个 token 的完整向量。

这里切分的是词表行，每个有效词条的隐藏维度仍然完整。

### 5.3. ParallelLMHead：Prefill 只投影每个请求的最后位置

`ParallelLMHead` 继承词表分片结构，但执行的是线性投影：

$$
\mathrm{logits}=XW_{\mathrm{vocab}}^{\mathsf T}
$$

Prefill 时，不需要为所有输入位置都生成词表 logits。代码先取每个请求本轮最后一个 token 的隐藏状态：

```python
if batch.is_prefill:
    indices = batch.attn_metadata.get_last_indices(bs)
    x = x[indices].contiguous()
```

例如两个请求本轮分别计算 3 个和 2 个 token，展平长度为 5，累计 Query 长度为 `[0, 3, 5]`，最后位置索引就是 `[2, 4]`。LM Head 只处理这两行，输出每个请求下一 token 的分数。

这一优化发生在输出头之前；Decoder 仍需处理本轮全部输入 token，以建立 KV 并计算正确的最后位置隐藏状态。

Decode 每个位置本来就对应一个请求的当前输入，因此不执行上述 Prefill 取行逻辑。图填充产生的额外 logits 由引擎按真实 `batch.size` 截取。

### 5.4. TP 下的 logits 聚合

每个 rank 先得到 `[B, V_local]` 的局部词表 logits，再通过 All-Gather 汇总。通信接口沿第 0 维拼接，因此结果需要从 rank 优先的布局重排成请求优先的布局。

```text
All-Gather 后：rank 0 的各请求 → rank 1 的各请求 → ...
重排后：      请求 0 的完整词表 → 请求 1 的完整词表 → ...
```

源码通过 `view`、`permute` 和 `reshape` 完成重排，最后切到实际词表大小。单请求有直接展平的快捷路径。

Embedding 用 All-Reduce 合并同一 token 的向量，LM Head 用 All-Gather 拼接不同词表区间的分数；两者的通信目的不同。

### 5.5. tie_word_embeddings：共享输入与输出权重

开启权重共享时，`lm_head.tied_embedding` 指向 `model.embed_tokens`。前向通过：

```python
module = self.tied_embedding or self
logits = F.linear(x, module.weight, self.bias)
```

实际使用输入 Embedding 的权重，而不是要求另外加载一份输出权重。共享模式下，LM Head 的 `state_dict()` 不重复导出权重，加载时也会消费 checkpoint 中可能存在的冗余 `lm_head.weight` 或 `lm_head.bias`。

当前模型构造输出头时没有启用 bias。共享靠前向选择同一权重对象实现，不是每次前向都复制输入权重到输出头。

### 5.6. 词表不能整除 TP 规模时的实现边界

Embedding 构造函数按向上取整的 `V_local` 分配每个 rank 的权重，但当前 `_shard_tensor()` 对词表只截取实际存在的行，没有补齐最后一个 rank 的 padding。

例如 `V=10`、`p=3` 时，最后一个 rank 的占位权重有 4 行，加载切片却只有 2 行，会与 `load_state_dict()` 的形状断言冲突。因此，不能仅凭 LM Head 最后有词表裁剪，就认为整个加载链已经完整支持所有不能整除的词表配置。

## 6. linear.py：张量并行与合并投影

### 6.1. _LinearTPImpl 的形状约定

底层权重按 PyTorch 的 `[out_features, in_features]` 存储，前向调用：

```python
F.linear(x, self.weight, self.bias)
```

也就是 $Y=XW^{\mathsf T}+b$。类中同时保存完整输入/输出维度和本 rank 的输入/输出维度，真正分配的张量使用局部维度。

讨论列并行、行并行时，最直接的方法是看切分了输入还是输出特征，不要仅根据二维 Tensor 的“行”“列”名称猜测。

### 6.2. LinearReplicated

该层在每个 rank 保存完整权重，输入与输出维度均不切分。当前 MoE 的路由器 `gate` 使用它，使各 rank 能依据相同隐藏状态计算完整专家集合的路由分数。

它自身没有额外的集体通信。

### 6.3. LinearColParallelMerged

这一层接收一个输入维度和多个输出维度，先分别计算每个输出分支的 TP 分片大小，再把局部分支合并为一个权重。

Dense MLP 的 gate 与 up 分支使用它：

```text
完整 gate 权重：[I, H]
完整 up 权重：  [I, H]

单 rank 合并权重：[2 × I/p, H]
单 rank 输出：    [T, 2 × I/p]
```

每个 rank 获得不同的中间特征，暂时不做 All-Gather；后续激活和 down 投影直接消费这些局部特征。

### 6.4. LinearQKVMerged

Q、K、V 同样合并为一次线性投影，但局部形状要分别计算 Query 和 KV head：

```python
local_num_qo = div_even(num_qo_heads, tp_info.size)
local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
```

Q head 要能被 TP 规模整除。KV head 数少于 TP 规模时，在满足复制条件的情况下，每个 rank 保存一个 KV head，并允许多个 rank 使用同一个全局 KV head。

例如 `num_qo_heads=8`、`num_kv_heads=2`、`head_dim=128`、`p=4`，每个 rank 有 2 个 Q head 和 1 个 KV head，局部 QKV 输出宽度为 `(2+1+1)×128=512`。

此时不是简单把完整 QKV 总宽度除以 4，因为 KV 分支发生了复制。

### 6.5. LinearOProj 与 LinearRowParallel

这两个实现都切分输入特征、保留完整输出特征。本 rank 先计算局部贡献，再做 All-Reduce 求和：

$$
Y=\sum_{r=0}^{p-1}X_rW_r^{\mathsf T}
$$

`LinearOProj` 接收本 rank 的 Attention head 输出，将它映射为 `[T, H]` 的局部贡献；`LinearRowParallel` 用于 MLP 的 down 投影，将本地中间特征映射回隐藏维度。

列并行与行并行配合后，中间特征可以一直留在本 rank，直到输出恢复隐藏维度时再通信。当前模型的这两类输出投影均设置 `has_bias=False`。

## 7. models/utils.py：Attention 的组合

### 7.1. RopeAttn 的成员

| 成员 | 作用 |
| --- | --- |
| `qkv_proj` | 一次线性运算产生本地 Q、K、V |
| `q_norm`、`k_norm` | 可选的逐 head RMSNorm |
| `attn` | 执行 QKV 拆分、RoPE 与 Attention 后端调用 |
| `o_proj` | 将本地 Attention 输出投影回隐藏维度并归约 |

`RopeAttn.forward()` 本身很短：

```python
qkv = self.qkv_proj.forward(x)
del x
o = self.attn.forward(qkv)
return self.o_proj.forward(o)
```

QKV 合并减少了独立投影调用，但没有改变 Q、K、V 各自的数学角色。`del x` 只是删除当前局部变量引用，不能据此推断张量的显存立即归还给系统。

### 7.2. AttentionLayer 的执行顺序

```python
q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
if self.q_norm is not None:
    self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
if self.k_norm is not None:
    self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
q, k = self.rotary.forward(ctx.batch.positions, q, k)
q = q.view(-1, self.num_qo_heads, self.head_dim)
o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
return o.view(-1, self.qo_attn_dim)
```

执行顺序是拆分 QKV、可选 QK Norm、RoPE、Attention。Q/K Norm 在 `head_dim` 上归一化，并且位于位置旋转之前；V 不经过这里的 QK Norm，也不应用 RoPE。

`num_qo_heads` 和 `num_kv_heads` 在这个算子内部表示本 rank 的数量。构造函数还要求全局 Q head 数能被 KV head 数整除，以符合分组共享的关系。

这里不直接实现历史 KV 的读取和 softmax，而是将 Q、K、V、层号和批次交给 `ctx.attn_backend`。后端负责按当前批次的页表、长度和阶段执行实际 Attention。

### 7.3. 与 KV Cache 的连接

位置编码使用 `ctx.batch.positions`，这是每个 token 在各自请求中的逻辑位置，不是整个拼接张量中的行号，也不是物理缓存槽位。

KV 写入则使用 `batch.out_loc`。每层的 `layer_id` 指定写入缓存池的哪一层，相同 token 在各层使用相同槽位编号，但保存不同层的 K、V。

后端写入本轮 K、V 后，再结合历史 KV 计算输出。模型层不负责申请页，页分配发生在调度器准备批次时；相关存储与生命周期细节见 [Mini-SGLang 解析（4）](/my-blog/posts/inference/minisgl_4/)。

## 8. rotary.py：RoPE 的预计算与复用

### 8.1. 基础频率与位置表

`RotaryEmbedding` 按如下方式构造逆频率：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
self._cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
```

频率可以写为 $\omega_i=\mathrm{base}^{-2i/d}$，位置 $m$ 对应旋转角度 $m\omega_i$。对一对参与旋转的分量，概念上的变换为：

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}
=
\begin{bmatrix}\cos(m\omega_i)&-\sin(m\omega_i)\\\sin(m\omega_i)&\cos(m\omega_i)\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
$$

缓存表形状为 `[max_position, rotary_dim]`，前半部分保存 cos，后半部分保存 sin。前向交给 FlashInfer 的 `apply_rope_with_cos_sin_cache_inplace()`，按位置索引对 Q、K 原地旋转。

`_cos_sin_cache` 是运行时预计算数据，不是 checkpoint 参数。当前实现要求 `rotary_dim == head_size`，并限制 head size 为 64、128、256 或 512。

### 8.2. _get_rope 的缩放分支

| 配置 | 当前源码的处理 |
| --- | --- |
| 没有 `rope_scaling` 或 `rope_type="default"` | 使用基础逆频率 |
| `rope_type="llama3"` | 按波长区间调整频率，在高低频区间之间进行平滑过渡 |
| `rope_type="yarn"` | 根据原位置长度和 beta 参数确定维度区间，用 ramp 混合原频率与缩放频率 |
| 其他类型 | 抛出 `ValueError` |

这里解析的是项目中已经实现的频率变换。尤其是 `yarn` 分支，不能仅凭名称推断它包含其他实现中所有可能的幅值缩放或配置选项；本文件展示的是 inverse frequency 的混合过程。

### 8.3. get_rope、缓存与 meta 初始化

`get_rope()` 使用 `functools.cache`，相同参数可以共享同一个 RoPE 对象，避免各 Decoder 层重复构建位置表。缩放字典在调用时转换为元组，以作为可哈希的缓存参数；其中的值也需要可哈希。

引擎在 `torch.device("meta")` 下创建模型，但位置表必须在真实设备上计算。`get_rope()` 检测到当前设备为 meta 时，会临时切换到 `_ROPE_DEVICE`；这个设备由引擎提前调用 `set_rope_device()` 设置。

因此，“模型在 meta 上创建”主要适用于等待 checkpoint 替换的权重，并不代表初始化阶段完全没有真实 GPU 张量分配。

## 9. GatedMLP 与 MoEMLP

### 9.1. GatedMLP 的计算

Dense 模型的 MLP 使用 gate 和 up 两个分支：

$$
\operatorname{MLP}(X)
=\left(\phi(XW_g^{\mathsf T})\odot XW_u^{\mathsf T}\right)W_d^{\mathsf T}
$$

其中 $\phi$ 是 SiLU 或 GELU，$\odot$ 是逐元素乘法。它不是简单的“两层线性层中间加一次激活”，因为 gate 的结果会与独立的 up 分支相乘。

代码先用 `gate_up_proj` 一次算出两路输出，再调用 `silu_and_mul()` 或 `gelu_and_mul()`，最后交给 `down_proj`。

`hidden_act` 当前只接受 `"silu"` 和 `"gelu"`，其他名称抛出 `ValueError`。激活包装函数位于 `layers/activation.py`，实际计算委托给 FlashInfer。

### 9.2. Dense MLP 的形状变化

设隐藏维度为 $H$、完整中间维度为 $I$、TP 规模为 $p$：

```text
输入                   [T, H]
gate_up_proj 输出      [T, 2I/p]
激活并逐元素相乘        [T, I/p]
down_proj 局部输出     [T, H]
All-Reduce 后          [T, H]
```

gate 与 up 必须使用一致的中间维度切分，才能在本 rank 上逐元素相乘。加载器因此需要分别切分两路权重，再按照 `[gate, up]` 的次序合并。

### 9.3. MoEMLP：路由器与专家层

MoE 版本用一个复制式线性层 `gate` 计算路由 logits，再交给 `MoELayer`：

```python
router_logits = self.gate.forward(hidden_states)
final_hidden_states = self.experts.forward(
    hidden_states=hidden_states, router_logits=router_logits
)
```

输入为 `[T, H]`，专家数量为 $E$ 时，路由 logits 形状为 `[T, E]`。每个 token 选择 `num_experts_per_tok` 个专家，并根据选中权重组合专家输出。

可以用示意公式表达为：

$$
y_t=\sum_{e\in S_t}\alpha_{t,e}\operatorname{Expert}_e(x_t)
$$

$S_t$ 是 token $t$ 的 Top-K 专家集合。`norm_topk_prob` 决定是否对选中的路由权重重新归一化。

当前 `fused` 后端先调用 `fused_topk()` 获取专家编号与权重，再执行融合专家计算。专家分组、计算块对齐等属于 `moe/fused.py` 的后端细节，模型层只提供路由分数与专家权重。

### 9.4. MoELayer 的权重与 TP

设专家中间维度为 $I_e$，每个 rank 的权重为：

```text
gate_up_proj：[E, 2I_e/p, H]
down_proj：   [E, H, I_e/p]
```

每个 rank 都保存全部专家编号，但只保存每个专家的一部分中间维度。专家计算产生局部 `[T, H]` 贡献后，再通过 All-Reduce 合并。

这是专家内部的张量并行，不能解释成“每个 rank 只负责某几个完整专家”的专家并行。当前这条路径也没有在模型层执行按专家分发的 All-to-All。

`MoEMLP` 没有把 `config.hidden_act` 传给 `MoELayer`，所以当前这条模型路径使用其默认 `activation="silu"`，不能直接套用 Dense MLP 的激活选择逻辑。

## 10. 各模型实现的差异

多个模型文件都采用 Embedding、Decoder 列表、最终 RMSNorm 和 LM Head 的骨架，主要差异集中在 Attention 参数和 MLP 类型。

| 模型 | QKV bias | Q/K Norm | MLP |
| --- | --- | --- | --- |
| Llama | 无 | 无 | GatedMLP |
| Qwen2 | 有 | 无 | GatedMLP |
| Qwen3 | 无 | 有 | GatedMLP |
| Qwen3 MoE | 无 | 有 | MoEMLP |
| 本项目的 Mistral 实现 | 无 | 无 | GatedMLP |

Qwen2 在创建 `RopeAttn` 时传入 `has_attn_bias=True`，只影响 QKV 投影；O 投影仍没有 bias。Qwen3 与 Qwen3 MoE 传入 `has_qk_norm=True`，因此添加每个 head 上的 Q/K RMSNorm。

Mistral 文件在当前版本中沿用与 Llama 相同的主要骨架。`ModelConfig` 没有保留 `sliding_window` 字段，这条模型调用链也没有向 Attention 传入滑动窗口参数，因此不能仅凭架构名称就推断已覆盖所有 Mistral 变体的窗口行为。

`Mistral3ForConditionalGeneration` 被映射到文本模型实现，加载器也会跳过指定视觉与投影权重；当前入口应理解为提取语言模型部分，而不是完整多模态推理。

## 11. weight.py：从 checkpoint 到运行时权重

### 11.1. load_weight 的整体流程

```text
定位或下载模型目录
    ↓
读取配置与 safetensors 文件
    ↓
规范化权重名称
    ↓
按 TP rank 切分单个张量
    ↓
合并 Q/K/V 或 gate/up
    ↓
MoE 权重按专家编号堆叠
    ↓
yield (运行时名称, 张量)
    ↓
引擎转换 dtype 并组装字典
    ↓
BaseOP.load_state_dict 替换占位权重
```

加载器扫描目录中的 `*.safetensors`。如果存在其他文件，会过滤名为 `consolidated.safetensors` 的文件；过滤后为空时才回退到原列表。

读取时跳过名称以 `vision_tower.`、`multi_modal_projector.` 开头的权重，并移除 `language_model.` 前缀。这里处理的是明确列出的命名模式，不是通用的多模态 checkpoint 解析规则。

### 11.2. _shard_tensor：让权重分片与层形状一致

| 权重类别 | 分片方式 |
| --- | --- |
| Q、K、V 投影 | 沿第 0 维，即输出特征切分；KV 可进入复制分支 |
| gate、up 投影 | 沿第 0 维，即中间特征切分 |
| O、down 投影 | 沿第 1 维，即输入特征切分 |
| Embedding、LM Head | 沿词表行切分 |
| Norm、MoE 路由器等其他权重 | 不在这些规则中切分 |

这些维度与 `F.linear` 的 `[out, in]` 存储约定一致。切片后调用 `clone()`，让局部分片拥有自己的存储，而不继续持有完整投影张量的底层空间。

KV head 少于 TP 规模时，复制分支按下面的式子确定本 rank 使用哪个完整 KV head：

```python
head_dim = value.shape[0] // num_kv_heads
head_idx = r * num_kv_heads // n
return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
```

例如 2 个 KV head、4 个 rank，rank 0 和 1 选择 head 0，rank 2 和 3 选择 head 1，与前面 `LinearQKVMerged` 和缓存池的局部 head 数相对应。

### 11.3. _get_merge_info：分别分片后再合并

checkpoint 常保存独立的 `q_proj`、`k_proj`、`v_proj`，而运行时使用一个 `qkv_proj`。加载器把各个分支暂存在 `merge_buf`，全部到齐后按固定顺序拼接：

```text
q_proj + k_proj + v_proj → qkv_proj，顺序为 [q, k, v]
gate_proj + up_proj     → gate_up_proj，顺序为 [gate, up]
```

代码使用 `torch.cat(parts, dim=0)`，名称中的投影部分也随之替换。

“先分别分片，再合并”的顺序很关键。如果先拼接完整 QKV，再对拼接张量做普通均匀切片，某个 rank 可能只拿到 Q 的一段，而不是它需要的局部 Q、K、V 三部分。

Qwen2 的投影 bias 也遵循对应分片与合并命名规则，最终成为 `qkv_proj.bias`。

### 11.4. _get_expert_stack_info：把独立专家装入批量张量

MoE checkpoint 常按专家编号保存：

```text
model.layers.0.mlp.experts.0.gate_proj.weight
model.layers.0.mlp.experts.0.up_proj.weight
model.layers.0.mlp.experts.1.gate_proj.weight
...
```

每个专家先完成本地 gate/up 合并，再通过正则表达式提取专家编号，将专家维度从名称移到张量第 0 维：

```text
运行时名称：model.layers.0.mlp.experts.gate_up_proj
张量形状：  [E, 2I_e/p, H]
```

运行时 `MoELayer.gate_up_proj` 本身就是 Tensor，所以打包名称不保留末尾 `.weight`。`expert_buf` 等待同一组的全部专家到齐，再按 `0 ... E-1` 的顺序调用 `torch.stack()`。

因此，专家数量和编号必须与配置匹配。文件读取结束时还会检查 `merge_buf` 和 `expert_buf` 是否为空，发现未完成的投影组或专家组就报错。

### 11.5. 流式加载的内存含义

`load_weight()` 使用迭代器逐组产出权重，但不能据此认为全程只有一个很小的临时张量。

当前 `safe_open()` 使用目标 `device`，读取后的完整张量、分片副本、等待其他分支的合并缓冲，以及等待完整专家集合的堆叠缓冲，都可能贡献设备内存峰值。具体临时量还与 checkpoint 的组织和读取顺序有关。

引擎最终会把产出的全部权重收集成字典，用于一次模型加载。因此，这里的流式主要描述读取与转换过程，模型最终仍需持有全部本 rank 权重；不能把文件注释中的简化内存描述理解为严格的峰值保证。

## 12. 用一个批次串起模型前向

假设采用不带 QK Norm 的 Dense 模型，两个请求本轮分别处理 3 个和 2 个 token，总计 `T=5`。设置 `H=1024`、8 个 Q head、2 个 KV head、`D=128`、`I=4096`、`TP=2`，并假设词表大小可被 2 整除。

| 阶段 | 每 rank 的主要张量形状 | 解释 |
| --- | --- | --- |
| 输入 token | `[5]` | 两个请求的本轮输入拼接 |
| Embedding | `[5, 1024]` | 局部查表后通过 All-Reduce 得到完整隐藏向量 |
| QKV 投影 | `[5, 768]` | 本地 4 个 Q head、1 个 K head、1 个 V head |
| 拆分 Q/K/V | `[5, 512]`、`[5, 128]`、`[5, 128]` | 随后对 Q、K 应用位置编码 |
| Attention 输出 | `[5, 512]` | 对应本地 4 个 Q head |
| O 投影与归约 | `[5, 1024]` | 恢复完整隐藏维度 |
| gate/up 合并投影 | `[5, 4096]` | 本地 gate 与 up 各有 2048 维 |
| 激活与逐元素乘法 | `[5, 2048]` | 本地中间特征 |
| down 投影与归约 | `[5, 1024]` | 作为 MLP 输出与残差继续传递 |
| 最后一层归一化 | `[5, 1024]` | 合并最后一次延迟的残差 |
| Prefill 取最后位置 | `[2, 1024]` | 取展平索引 2 和 4 |
| LM Head 与聚合 | `[2, vocab_size]` | 每个请求一组完整词表分数 |

Attention 读取的历史长度可以大于本轮这 5 个 token，因为命中或此前计算好的 KV 已在缓存中。线性层和 MLP 只处理本轮输入，Attention 后端通过元数据连接当前 Query 与完整历史。

当引擎拿到 logits 后，再调用采样器选择两个新 token。下一轮 Decode 通常只将这两个新 token 送入同一个模型入口；模型没有另一套独立的 Decode 网络，差别在批次输入、Attention 元数据以及是否使用 CUDA Graph。

本文核对了 `models/`、`layers/` 的实现，并结合 `engine/engine.py`、`distributed/impl.py`、`attention/fa.py`、`attention/fi.py`、`moe/fused.py` 和 `kernel/csrc/jit/index.cu` 检查直接调用关系。继续阅读 Attention 后端时，可以重点跟踪 `AttentionLayer.forward()` 之后，Q、K、V 如何与缓存页表和序列边界共同进入计算内核。
