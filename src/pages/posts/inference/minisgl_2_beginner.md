---
layout: ../../../layouts/PostLayout.astro
category: inference
title: Mini-SGLang 解析（2）：初学者学习版
description: 跟着一个批次读懂 Mini-SGLang，从 token 到 logits，再理解张量并行、模型创建与权重加载
date: 2026-9-13
---
# Mini-SGLang 解析（2）：初学者学习版

在 [Mini-SGLang 解析（1）](/my-blog/posts/inference/minisgl_1/) 中，引擎执行了 `self.model.forward()`，随后根据返回的 logits 采样新 token。这次我们把这个调用展开，看看输入的 token 如何一步步变成下一个 token 的分数。

本文适合刚开始阅读推理框架源码的读者。只需要知道 Tensor 有形状、矩阵可以相乘，以及模型会重复执行多层计算。第一次阅读时，可以先读到第 7 节，走完单卡前向；再读后面的多卡执行与初始化。

这是[原版解析（2）](/my-blog/posts/inference/minisgl_2/)的独立学习版，依据原文所对应的 Mini-SGLang 版本 `d0c2ecefca4ade346fb11d0d4ef1191ae73efdbc` 整理。源码路径均相对于 Mini-SGLang 项目根目录；标为“示意”的代码只用于解释数据流，完整字段、实现边界与更多细节可回到原文查阅。

## 1. 先明确：模型这一轮要完成什么

假设有两个请求，A 本轮要处理 3 个 token，B 要处理 2 个 token。调度器把它们组织成一个批次，模型看到的输入可以表示为：

```text
请求 A：a0 a1 a2
请求 B：b0 b1

batch.input_ids = [a0, a1, a2, b0, b1]
形状：[5]
```

这里的 `a0` 等符号代表 token ID，也就是词表中的整数编号。我们把本轮 token 总数记为 `T`，隐藏维度记为 `H`，词表大小记为 `V`。这个例子里 `T=5`，并假设 `H=1024`。这些数字只是为了方便跟踪形状。

对于这轮 Prefill，模型最后要返回两组词表分数：一组用于预测 A 的下一个 token，一组用于预测 B 的下一个 token。因此，最终 logits 的形状是 `[2, V]`。logits 是采样前的分数，不是已经选好的 token，也不必是概率。

中间过程先记住下面这条路线：

```text
token ID          [5]
    ↓ Embedding：把编号换成向量
隐藏状态           [5, 1024]
    ↓ 多个 Decoder Layer：结合上下文，更新向量
隐藏状态           [5, 1024]
    ↓ 最终归一化，取每个请求的最后位置
用于预测的隐藏状态  [2, 1024]
    ↓ LM Head：计算词表中每个 token 的分数
logits            [2, V]
    ↓ 引擎中的采样器
两个新 token
```

五个 token 放在同一个张量里，并不代表两个请求会互相看到内容。Attention 使用批次元数据识别请求边界、序列长度和缓存位置。普通线性层则可以对这五行一起计算。

## 2. 从 llama.py 看见完整前向

先打开 `python/minisgl/models/llama.py`。Llama 的结构适合作为起点：先理解这个模型，再看其他模型增加了什么。

最外层 `LlamaForCausalLM.forward()` 的核心只有两步：

```python
output = self.model.forward(get_global_ctx().batch.input_ids)
logits = self.lm_head.forward(output)
return logits
```

`self.model` 负责生成隐藏状态，`self.lm_head` 负责把隐藏状态变成词表分数。输入从当前 `Context` 的 `batch` 中取得，这是第一篇里引擎进入批次上下文后设置好的。

继续进入 `LlamaModel.forward()`：

```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    x = self.embed_tokens.forward(input_ids)
    residual: torch.Tensor | None = None
    for layer in self.layers.op_list:
        x, residual = layer.forward(x, residual)
    return self.norm.forward(x, residual)[0]
```

现在可以把刚才的路线与成员名称对应起来：`embed_tokens` 是输入端，`layers` 是反复执行的 Decoder 层，`norm` 是最后的归一化。`residual` 保存残差，下一节之后再展开。

阅读这里时，先沿着 `forward()` 跟踪张量。构造函数里配置如何传递、这些类为什么继承 `BaseOP`，等走完前向后再回头看，会更容易理解它们的用途。

## 3. Embedding：把 token 编号换成向量

在 `python/minisgl/layers/embedding.py` 中找到 `VocabParallelEmbedding`。先考虑单卡情形，把它理解成一张形状为 `[V, H]` 的表：每个 token ID 对应一行长度为 `H` 的向量。

```text
输入编号：      [a0, a1, a2, b0, b1]     [5]
按编号取出向量： [向量, 向量, 向量, 向量, 向量]  [5, 1024]
```

这张表的数值来自训练好的权重。查表之后，后续层就能对浮点向量进行计算。

此时每一行主要表示对应 token 的输入特征；它与同一请求中其他位置的信息如何结合，要到 Attention 中才发生。Embedding 本身不负责读取历史 KV Cache。

类名中的 `Parallel` 表示它也支持把词表分到多张卡上。暂时按单卡查表理解即可，第 8 节再解释各卡如何共同得到完整向量。

## 4. Decoder Layer：先理解一层怎样更新隐藏状态

回到 `llama.py`，找到 `LlamaDecoderLayer.forward()`：

```python
def forward(self, x, residual=None):
    x, residual = self.input_layernorm.forward(x, residual)
    x = self.self_attn.forward(x)
    x, residual = self.post_attention_layernorm.forward(x, residual)
    x = self.mlp.forward(x)
    return x, residual
```

一层里有两个主要计算模块：Attention 让当前位置结合上下文，MLP 对每个位置的特征做进一步变换。两个模块之前都有归一化，同时通过残差连接保留并累加已有信息。

先把这一层写成更熟悉的数学步骤：

```python
# 示意：展开残差相加，便于理解数学关系
a = attention(norm1(h))
r = h + a
m = mlp(norm2(r))
h_next = r + m
```

输入与输出都具有 `[T, H]` 的形状，所以可以把这样的层串联多次。下面先解决源码与这个示意写法之间的差别，再进入 Attention 和 MLP。

### 4.1. RMSNorm 做了什么

RMSNorm 根据一个向量的均方根调整数值尺度，再乘上一组可学习权重。对长度为 `H` 的向量 $x$：

$$
\operatorname{RMSNorm}(x)_i
= w_i\frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2+\epsilon}}
$$

分母根据当前向量计算，$\epsilon$ 用于数值稳定，$w_i$ 来自模型权重。它不改变张量形状，也不执行减均值。对应实现把计算交给 FlashInfer 的归一化算子。

### 4.2. 源码里的残差加法在哪里

`RMSNormFused` 把残差相加与归一化放到一起执行。第一次调用时 `residual` 为 `None`，它保留原始输入作为 residual，并返回归一化后的输入。

之后每次调用，会先把 `x` 加进 `residual`，再得到归一化结果。于是，一层完成 MLP 后返回的是刚才示意代码中的 `(m, r)`，最后的 `r + m` 留到下一次归一化时完成。

```text
当前层返回：             x = m，residual = r
下一层 input_layernorm：  先合并 r + m，再归一化
如果已是最后一层：        由 LlamaModel 末尾的 norm 完成合并
```

因此，跟踪层间状态时要同时看 `x` 和 `residual`。只看 `x`，会误以为上一层留下的信息消失了。

## 5. Attention：让当前位置结合上下文

现在进入 `python/minisgl/models/utils.py` 中的 `RopeAttn`。它把 Attention 组织为三步：

```python
qkv = self.qkv_proj.forward(x)
o = self.attn.forward(qkv)
return self.o_proj.forward(o)
```

这三步分别是准备 Q/K/V、计算 Attention、将结果映射回隐藏维度。

### 5.1. 为什么先计算 Q、K、V

可以把 Query 理解为当前位置用来寻找信息的表示，Key 是用于匹配的表示，Value 是匹配后要汇总的内容。三者都由输入隐藏状态经过线性投影得到。

这里的“线性投影”就是带权重的矩阵乘法。项目使用的 `F.linear` 按 `[输出维度, 输入维度]` 保存权重，执行 $Y=XW^{\mathsf T}+b$；Llama 的 QKV 投影没有 bias。

Mini-SGLang 将三份投影合并成 `qkv_proj`，一次计算后再拆出 Q、K、V。合并改变了执行方式，但三者仍有各自的用途。

Attention 会分成多个 head。继续使用 `T=5`，假设有 8 个 Q head、2 个 KV head，每个 head 的维度 `D=128`，在单卡上：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| 输入 | `[5, 1024]` | 五个 token 的隐藏状态 |
| 合并 QKV | `[5, 1536]` | 总宽度为 `(8 + 2 + 2) × 128` |
| Q | `[5, 1024]` | 每个 token 有 8 个 Query head |
| K、V | 各 `[5, 256]` | 每个 token 各有 2 个 KV head |

多个 Q head 共享较少的 KV head，这种分组关系可以减少 KV 数据量。源码要求 Q head 数能被 KV head 数整除。

### 5.2. RoPE：在计算注意力前加入位置信息

`AttentionLayer` 拆开 QKV 后，对 Q 和 K 应用 RoPE，再调用 Attention 后端。Llama 这条路径没有 Q/K Norm；Qwen3 等模型会先做逐 head 归一化，再做 RoPE。

RoPE 根据 token 在请求中的位置旋转 Q、K 的分量，使后续匹配能够包含位置信息。V 不参与这里的旋转。

位置来自 `ctx.batch.positions`。如果示例中两个请求都从位置 0 开始且没有缓存前缀，那么：

```text
拼接行号： 0  1  2  3  4
逻辑位置： 0  1  2  0  1
所属请求： A  A  A  B  B
```

请求 B 的第一个 token 虽然位于拼接张量第 3 行，但它在自己的请求中仍是位置 0。

需要继续追踪实现时，再打开 `python/minisgl/layers/rotary.py`：它预计算不同位置使用的 cos/sin 表，前向时按位置取值进行旋转；`get_rope()` 还会复用相同配置的对象。第一次读到这里，理解“Q/K 在匹配前经过位置变换”即可。

### 5.3. Attention 如何接上历史 KV

`AttentionLayer` 将 Q、K、V、当前层号和批次交给 `ctx.attn_backend.forward()`。真正的历史缓存访问和 Attention 计算在后端完成。

本轮 K、V 通过 `batch.out_loc` 对应到缓存位置，`layer_id` 指定模型的哪一层。后端再结合请求边界、序列长度和历史 KV，为当前 Query 计算输出。

这解释了为什么 Decode 可以只输入最新 token：前面位置的 K、V 已经保存在缓存里，本轮 Query 仍然可以使用历史信息。Attention 访问的历史长度可能远大于本轮输入行数。

缓存页在调度器准备批次时分配。有关页表和缓存生命周期，可以继续读 [Mini-SGLang 解析（4）](/my-blog/posts/inference/minisgl_4/)。

### 5.4. O 投影把结果送回残差流

Attention 的输出对应各个 Q head。`o_proj` 将拼接后的 head 输出映射回 `H` 维，让它能与 `[T, H]` 的残差相加。

示例中 `8 × 128 = 1024`，Attention 输出宽度恰好等于隐藏维度，但这不是所有配置都必须满足的关系。QKV 与 O 投影负责衔接隐藏维度和 Attention 维度。

## 6. MLP：对每个 token 的特征继续加工

完成 Attention 后，Decoder 合并残差并归一化，再调用 `models/utils.py` 中的 `GatedMLP`。

MLP 分别处理每个 token 的隐藏向量。在这一步，不同 token 之间没有像 Attention 那样的信息汇总。

这个 MLP 包含 gate、up 和 down 三份投影：gate 与 up 从同一个输入出发，gate 经过激活后与 up 逐元素相乘，最后 down 将结果变回隐藏维度。

```python
# 示意：展示分支关系，实际实现合并了 gate/up 投影
gate = activation(linear_gate(x))
up = linear_up(x)
x = linear_down(gate * up)
```

设中间维度 `I=4096`，单卡上的形状变化是：

```text
输入                        [5, 1024]
gate_up_proj 合并输出        [5, 8192]
拆成 gate、up               各 [5, 4096]
激活 gate，再与 up 逐元素相乘 [5, 4096]
down_proj 输出              [5, 1024]
```

`layers/activation.py` 中的包装函数把“激活并相乘”交给 FlashInfer。Dense MLP 当前支持 SiLU 和 GELU 两种激活。

MLP 输出会与 residual 一起返回，交给下一层继续处理。经过全部 Decoder 层后，模型末尾的 `norm` 合并最后一次残差并归一化。此时五行隐藏状态已经包含各自允许访问的上下文信息，可以交给输出头了。

## 7. LM Head：从隐藏状态得到下一个 token 的分数

回到 `layers/embedding.py`，这次看 `ParallelLMHead`。虽然它与 Embedding 写在同一个文件中，但现在我们处于模型输出端。

LM Head 使用形状为 `[V, H]` 的权重，计算：

$$
\mathrm{logits}=XW_{\mathrm{vocab}}^{\mathsf T}
$$

每一行隐藏状态会得到 `V` 个分数，分别对应词表中的一个候选 token。

### 7.1. 为什么五个输入位置只产生两组分数

这轮 Prefill 要为两个请求各预测一个新 token，所以只需要每个请求最后一个输入位置的隐藏状态。

```text
隐藏状态的行： a0  a1  a2  b0  b1
取出索引：            2       4

选取前：[5, 1024]
选取后：[2, 1024]
投影后：[2, V]
```

代码通过 `batch.attn_metadata.get_last_indices(bs)` 得到这些索引，在词表投影之前取行。Decoder 仍需处理本轮全部输入，以建立 KV 并算出正确的最后位置隐藏状态。

如果配置启用 `tie_word_embeddings`，输出头就使用输入 Embedding 的同一份权重：输入时按编号查行，输出时与这些词表向量做线性投影。这个开关是否启用取决于模型配置。

### 7.2. 模型返回之后，谁选择新 token

引擎取得 logits 后交给采样器，才得到两个新 token。请求是否停止，由后续流程根据长度、EOS 等条件判断。

下一轮 Decode 通常只把这两个新 token 作为本轮输入，因此 `T` 从 5 变为 2；它们通过同样的 Embedding、Decoder 和 LM Head。Attention 后端利用已有 KV 补足历史信息。使用 CUDA Graph 时可能还有填充位置，引擎会按真实批次大小截取 logits。

到这里，可以重新读一次 `LlamaModel.forward()`：查表、反复更新隐藏状态、最终归一化，再由外层输出头产生分数。接下来再解释源码里频繁出现的并行类名。

## 8. 多张 GPU 怎样共同执行这条前向

张量并行（Tensor Parallelism，简称 TP）把同一层的一部分权重和计算分给不同 GPU。下面把参与计算的进程称为 rank，TP 规模记为 `p`。

阅读 `python/minisgl/layers/linear.py` 时，重点看每个 rank 保存哪些特征，以及什么时候需要通信。

### 8.1. 从 MLP 理解“分开算，再相加”

把刚才 `I=4096` 的中间维度分给两个 rank，每个 rank 计算其中 2048 维。gate 与 up 必须切分对应的特征，才能在本 rank 上相乘。

```text
两个 rank 都有输入 [5, 1024]
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
rank 0：中间 2048 维   rank 1：中间 2048 维
       ↓ down 投影          ↓ down 投影
局部贡献 [5, 1024]     局部贡献 [5, 1024]
       └─────────┬─────────┘
                 ↓ All-Reduce 求和
       两个 rank 都得到完整 [5, 1024]
```

`LinearColParallelMerged` 用于 gate/up：切分输出特征，并把本地两路投影合并。`LinearRowParallel` 用于 down：消费本地中间特征，计算对完整隐藏向量的贡献，最后求和。

这里的 All-Reduce 可以理解为“把各 rank 的对应元素相加，并把结果交给每个 rank”。中间特征无需先全部拼回一张卡，down 投影就能直接使用本地结果。

### 8.2. Attention 按 head 分工

`LinearQKVMerged` 合并本 rank 需要的 Q、K、V 投影；`LinearOProj` 再把局部 Attention 输出投影成 `[T, H]` 的贡献，并通过 All-Reduce 求和。

仍用 8 个 Q head、2 个 KV head、`D=128`，当 `p=2` 时，每个 rank 有 4 个 Q head、1 个 K head、1 个 V head。因此，每个 rank 的 QKV 输出形状是 `[5, 768]`，宽度为 `(4+1+1)×128`。

KV head 少于 rank 数时，还可能发生复制。例如相同配置采用 `p=4`，每个 rank 有 2 个 Q head 和 1 个 KV head，部分 rank 共享同一个全局 KV head。此时不能直接把完整 QKV 宽度除以 4 来得到局部宽度；具体还要满足实现中的整除与复制条件。

### 8.3. Embedding 与 LM Head 为什么通信方式不同

词表也可以按行分给不同 rank。假设 `V=8`、`p=2`，rank 0 保存 token 0～3 的向量，rank 1 保存 token 4～7 的向量。

输入 token 5 时，rank 0 查表得到零向量，rank 1 查到 token 5 的向量。All-Reduce 求和后，两者就都有完整输入向量。

LM Head 则让各 rank 计算自己负责的词表分数：rank 0 得到候选 0～3 的分数，rank 1 得到候选 4～7 的分数。它需要通过 All-Gather 收集并拼接不同区间，再整理成每个请求的一行完整 logits。

| 位置 | 各 rank 手中的内容 | 需要的操作 |
| --- | --- | --- |
| Embedding 输出 | 本地命中的向量，其他位置为零 | All-Reduce 求和 |
| O/down 投影输出 | 对同一隐藏向量的局部贡献 | All-Reduce 求和 |
| LM Head 输出 | 不同词表区间的候选分数 | All-Gather 拼接并重排 |

上述词表示例都假设可整除。原文所对应版本中，词表分片的占位形状采用向上取整，但加载切片未补齐最后一片；不能据此假设任意词表大小都能直接用于任意 TP 规模。

## 9. 理解 Llama 后，再看其他模型改了哪里

现在阅读 `models/qwen2.py`、`qwen3.py`、`qwen3_moe.py`，可以先比较它们构造 Attention 和 MLP 的地方：

| 实现 | 相对于本文 Llama 主线的主要变化 |
| --- | --- |
| Qwen2 | QKV 投影增加 bias |
| Qwen3 | Q、K 在 RoPE 前增加逐 head RMSNorm |
| Qwen3 MoE | 保留 Q/K Norm，并用 MoEMLP 替换 Dense MLP |
| 本项目的 Mistral | 主要骨架沿用与 Llama 相同的组织方式 |

MoE 可以先理解为“有多组专家 MLP，每个 token 只选择其中几组”。`MoEMLP` 先通过路由器 `gate` 计算专家分数，再由 `MoELayer` 选择专家并组合输出。

假设有 `E` 个专家，路由分数形状就是 `[T, E]`。每个 token 选择的专家数由 `num_experts_per_tok` 指定。这里路由器的 `gate` 用于选择专家，与 Dense MLP 中参与逐元素相乘的 gate 分支用途不同。

路由器使用 `LinearReplicated`，每个 rank 保存完整权重。在当前 TP 实现中，每个 rank 都持有全部专家的一部分中间维度，专家输出贡献再求和。它没有把不同的完整专家分别放到不同 rank 上。

模型名称也需要结合实际实现理解。例如当前 Mistral 调用链没有传入滑动窗口参数；映射到文本实现的 `Mistral3ForConditionalGeneration` 入口也不代表完整的多模态前向。这些差异与 MoE 权重形状可以在原版中进一步查阅。

## 10. 回到启动阶段：这些层是怎样创建出来的

前面一直假设模型已经准备好。现在回到第一篇的引擎初始化，看看如何创建这条计算链。

先读 `python/minisgl/models/config.py`。`ModelConfig.from_hf()` 将 Hugging Face 配置整理成内部字段，其中最容易和前向对应的是：

| 配置字段 | 前面已经见过的用途 |
| --- | --- |
| `num_layers` | Decoder 重复多少层 |
| `hidden_size` | 隐藏状态宽度 `H` |
| `vocab_size` | 输入查表与输出候选数 `V` |
| `num_qo_heads`、`num_kv_heads`、`head_dim` | QKV 投影与拆分形状 |
| `intermediate_size` | Dense MLP 的中间宽度 `I` |
| `rotary_config` | RoPE 的位置表和频率配置 |
| `tie_word_embeddings` | 输入表和输出头是否共享权重 |

配置记录完整模型的维度，各算子再根据 TP 规模计算局部形状。`head_dim` 可以显式给出，因此不要无条件把 `hidden_size` 看成 Q head 数与 head 维度的乘积。

然后读 `models/__init__.py` 与 `models/register.py`。`create_model()` 使用 `architectures` 中的第一个名称查注册表，导入对应实现，再创建模型实例。

```text
Hugging Face 配置
    ↓ ModelConfig.from_hf()
统一的模型配置
    ↓ create_model() 按架构名查注册表
LlamaForCausalLM 等模型实例
    ↓ 构造 Embedding、Decoder、LM Head
具有所需权重形状的模型对象
```

学习顺序与启动顺序在这里交汇了：先看懂每一层做什么，再看配置怎样决定层数与维度。模型对象创建好后，还需要把训练得到的数值装入这些权重。

## 11. 权重加载：让文件中的参数对上运行时结构

### 11.1. 先看模型怎样给权重命名

打开 `python/minisgl/layers/base.py`。项目的 `BaseOP` 使用自己的状态管理逻辑，权重是普通 Tensor，而不是依赖 `torch.nn.Module` 的参数注册。

`state_dict()` 从公开属性收集 Tensor，遇到子 `BaseOP` 就继续递归。例如：

```text
model → layers → 0 → self_attn → qkv_proj → weight

model.layers.0.self_attn.qkv_proj.weight
```

Decoder 列表使用 `OPList`，让每一层可以按 `0、1、2……` 进入权重路径。以下划线开头的属性不参与这种收集；`StateLessOP` 则不参与 checkpoint 状态读写。

这样一来，模型对象的属性路径就定义了加载器最终需要生成的名称。

### 11.2. 为什么 checkpoint 不能直接原样装入

接着看 `python/minisgl/models/weight.py`。checkpoint 可能分别保存 `q_proj`、`k_proj`、`v_proj`，而我们刚才看到运行时使用一个 `qkv_proj`。

加载器需要把保存形式转换成执行形式，多卡时还要先取本 rank 的部分：

```text
读取 q_proj、k_proj、v_proj
    ↓ 分别取本 rank 所需的分片
局部 Q 权重、局部 K 权重、局部 V 权重
    ↓ 按 [Q, K, V] 沿输出维度拼接
qkv_proj.weight
```

必须先分别分片，再合并。若把完整 QKV 拼起来后直接均匀切段，一个 rank 可能只拿到 Q 的一部分，却缺少自己的 K、V。

MLP 的 gate/up 同样按 `[gate, up]` 合并。O/down 投影则切分输入特征，与第 8 节的计算分工对应。MoE 还需要按专家编号将独立专家权重堆叠成带专家维度的 Tensor。

### 11.3. 最后把占位张量替换成真实权重

引擎先在 meta 设备下构造权重的形状与类型占位，再加载真实设备权重。这里的 meta 可以理解为“记录张量应该长什么样，还没有实际的权重数值存储”。RoPE 位置表等运行时数据另有真实设备初始化逻辑。

加载器产出转换后的名称与 Tensor，引擎统一 dtype 并组成字典，最后调用模型的 `load_state_dict()`。

这个方法会检查名称、形状和 dtype，然后将相应属性替换为真实 Tensor。它不会向 meta 占位存储中复制数据。遗漏或多出的权重键会报错，形状和类型也必须匹配。

现在启动流程与前向计算就接上了：配置确定模型结构，构造函数准备权重形状，加载器完成名称和分片转换，模型随后使用这些权重处理批次。

## 12. 再读源码时，按这条路线走

可以按下面的顺序做第二遍阅读。每一步先回答对应问题，再继续进入下一层实现。

| 阅读顺序 | 文件或入口 | 需要回答的问题 |
| --- | --- | --- |
| 1 | `models/llama.py` 的两层模型 `forward()` | 输入从哪里来，隐藏状态怎样到达输出头？ |
| 2 | `layers/embedding.py` 的 `VocabParallelEmbedding` | token ID 怎样成为 `[T, H]` 的向量？ |
| 3 | `LlamaDecoderLayer` 与 `RMSNormFused` | Attention、MLP 和残差怎样衔接？ |
| 4 | `models/utils.py` 的 `RopeAttn`、`AttentionLayer`，再到 `layers/rotary.py` | QKV、位置和历史缓存怎样进入 Attention？ |
| 5 | `GatedMLP` 与 `layers/activation.py` | gate/up 如何合并，中间形状怎样变化？ |
| 6 | `layers/embedding.py` 的 `ParallelLMHead` | 为什么 Prefill 只投影每个请求的最后位置？ |
| 7 | `layers/linear.py`，再回看词表并行 | 各 rank 分别算什么，何时求和或拼接？ |
| 8 | 其他模型文件与 `MoEMLP` | 它们对已理解的骨架做了哪些改动？ |
| 9 | `models/config.py`、`register.py`、`__init__.py` | 配置如何变成具体模型对象？ |
| 10 | `layers/base.py` 与 `models/weight.py` | 权重文件怎样对应模型属性和局部形状？ |

阅读后可以用三个问题检查理解：为什么五行 Prefill 输入只需要两行 logits？为什么 Decoder 要同时返回 `x` 和 `residual`？为什么 QKV 权重要先分片再合并？分别能从输出位置、延迟残差相加和多卡分工解释清楚，就可以继续深入[原版解析（2）](/my-blog/posts/inference/minisgl_2/)中的配置细节、RoPE 缩放和加载边界了。
