---
layout: ../../../layouts/PostLayout.astro
category: papers-algorithms
title: 公式推导：多头注意力
description: 从加权求和出发，逐步推导缩放点积注意力与多头注意力的计算公式，并核对每一步的矩阵维度。
date: 2026-9-11
---
一、基于位置的加权求和

假设输入序列有 $n$ 个 token，每个 token 用一个 $d_{\text{model}}$ 维的行向量表示。把这些向量按行堆起来，得到：

$$
X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
\in \mathbb{R}^{n \times d_{\text{model}}}
$$

为了专注于注意力计算，下面先省略 batch 维度、投影偏置和 dropout。$X$ 表示进入注意力层的隐藏状态，位置编码等处理也不在这里展开。

对于第 $i$ 个位置，我们希望它能够根据当前输入，从其他位置读取信息。最直接的表达方式是加权求和：

$$
z_i = \sum_{j=1}^{n} a_{ij}v_j
$$

其中：

- $v_j$ 是第 $j$ 个位置提供的内容向量。
- $a_{ij}$ 表示第 $i$ 个位置分配给第 $j$ 个位置的权重。
- $z_i$ 是第 $i$ 个位置聚合信息后的输出。

我们希望权重满足：

$$
a_{ij} \geq 0, \qquad \sum_{j=1}^{n} a_{ij} = 1
$$

于是，问题变成了：**怎样根据输入，计算每个位置应该分配给其他位置的权重？**

## 二、用 Q、K、V 区分查询、匹配和内容

先把输入映射成三组向量：

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V
$$

投影矩阵的维度为：

$$
W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k},
\qquad
W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

因此：

$$
Q, K \in \mathbb{R}^{n \times d_k},
\qquad V \in \mathbb{R}^{n \times d_v}
$$

可以把 $q_i$ 理解为当前位置用来发起查询的表示，把 $k_j$ 理解为另一个位置用于匹配的表示，而 $v_j$ 是匹配后实际读取的内容。

这里 $Q$ 和 $K$ 的最后一维必须相同，因为后面要做点积；$V$ 的最后一维可以不同，因为它参与的是加权求和。

对位置 $i$ 和位置 $j$，先用点积定义匹配分数：

$$
s_{ij} = q_i k_j^{\top}
= \sum_{r=1}^{d_k} q_{ir}k_{jr}
$$

把所有位置两两之间的分数放在一起，就是：

$$
S = QK^{\top} \in \mathbb{R}^{n \times n}
$$

**$S$ 的行对应查询位置，列对应被读取的位置。** 这也是后面判断 softmax 应该沿哪个方向计算的依据。

## 三、为什么要除以根号 d_k

点积包含 $d_k$ 项相加。维度增大时，分数的波动可能随之增大。

为了估计这个尺度，我们做一个简化假设：$q$ 和 $k$ 的所有分量相互独立，均值为 $0$，方差为 $1$。对任意一个分量，有：

$$
\mathbb{E}[q_r k_r]
= \mathbb{E}[q_r]\mathbb{E}[k_r] = 0
$$

以及：

$$
\begin{aligned}
\operatorname{Var}(q_r k_r)
&= \mathbb{E}[q_r^2 k_r^2] - \mathbb{E}[q_r k_r]^2 \\
&= \mathbb{E}[q_r^2]\mathbb{E}[k_r^2] \\
&= 1
\end{aligned}
$$

由于不同分量的乘积也相互独立：

$$
\operatorname{Var}\left(\sum_{r=1}^{d_k} q_r k_r\right)
= \sum_{r=1}^{d_k}\operatorname{Var}(q_r k_r)
= d_k
$$

因此，未缩放点积的标准差是 $\sqrt{d_k}$。除以这个尺度后：

$$
\operatorname{Var}\left(\frac{qk^{\top}}{\sqrt{d_k}}\right)
= \frac{1}{d_k}\operatorname{Var}(qk^{\top})
= 1
$$

于是采用缩放后的分数：

$$
\widetilde{s}_{ij} = \frac{q_i k_j^{\top}}{\sqrt{d_k}}
$$

这个缩放有助于避免分数差异随维度增大而变得过大，使 softmax 过于尖锐、进入梯度很小的区域。

这里的独立性和单位方差是解释缩放动机的近似假设，训练中的 $Q$、$K$ 并不保证满足它们。**缩放控制的是分数尺度，不是把点积变成余弦相似度。**

## 四、把分数变成权重，再读取 V

固定查询位置 $i$，沿着所有 key 的位置做 softmax：

$$
a_{ij}
= \frac{\exp(\widetilde{s}_{ij})}
{\sum_{\ell=1}^{n}\exp(\widetilde{s}_{i\ell})}
$$

分子非负，分母是这一行所有分子的和，所以自然满足：

$$
a_{ij} \geq 0, \qquad \sum_{j=1}^{n}a_{ij}=1
$$

令 $A$ 为所有权重组成的矩阵，有：

$$
A = \operatorname{softmax}_{\text{row}}
\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)
\in \mathbb{R}^{n \times n}
$$

把第一节的加权求和写成矩阵形式：

$$
Z = AV \in \mathbb{R}^{n \times d_v}
$$

因为矩阵乘法的第 $i$ 行恰好是：

$$
(AV)_{i,:} = \sum_{j=1}^{n} A_{ij}V_{j,:}
= \sum_{j=1}^{n} a_{ij}v_j
$$

代入 $A$，就得到**缩放点积注意力**：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}_{\text{row}}
\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
}
$$

注意，softmax 的结果只是权重矩阵；乘上 $V$ 之后，才得到聚合后的内容。

## 五、从单头扩展到多头

单个注意力头为每个查询位置生成一组权重。我们可以让同一个输入经过多组不同的投影，分别计算注意力，再把结果组合起来。

设一共有 $h$ 个头。对第 $r$ 个头，定义：

$$
Q_r = XW_r^Q, \qquad
K_r = XW_r^K, \qquad
V_r = XW_r^V
$$

其中：

$$
W_r^Q,W_r^K \in \mathbb{R}^{d_{\text{model}}\times d_k},
\qquad W_r^V \in \mathbb{R}^{d_{\text{model}}\times d_v}
$$

每个头独立计算自己的权重矩阵和输出：

$$
A_r = \operatorname{softmax}_{\text{row}}
\left(\frac{Q_rK_r^{\top}}{\sqrt{d_k}}\right)
$$

$$
H_r = A_rV_r \in \mathbb{R}^{n\times d_v}
$$

将投影公式代入：

$$
H_r = \operatorname{softmax}_{\text{row}}
\left(\frac{(XW_r^Q)(XW_r^K)^{\top}}{\sqrt{d_k}}\right)XW_r^V
$$

不同头拥有不同的可学习参数，因此可以产生不同的匹配分数和信息聚合方式。我们并没有预先规定某个头负责语法、某个头负责语义，也不保证训练后每个头都有清晰的分工。

### 拼接各个头的输出

沿特征维度拼接，序列长度 $n$ 不变：

$$
H = \operatorname{Concat}(H_1,\ldots,H_h)
\in \mathbb{R}^{n\times hd_v}
$$

再引入输出投影：

$$
W^O \in \mathbb{R}^{hd_v\times d_{\text{model}}}
$$

得到：

$$
Y = HW^O \in \mathbb{R}^{n\times d_{\text{model}}}
$$

**多头注意力的计算公式由此写成：**

$$
\boxed{
\operatorname{MHA}(X)
= \operatorname{Concat}(H_1,\ldots,H_h)W^O
}
$$

其中：

$$
\boxed{
H_r = \operatorname{softmax}_{\text{row}}
\left(\frac{(XW_r^Q)(XW_r^K)^{\top}}{\sqrt{d_k}}\right)XW_r^V
}
$$

### 输出投影怎样融合各个头

把 $W^O$ 沿行分成 $h$ 块，每块记为 $O_r\in\mathbb{R}^{d_v\times d_{\text{model}}}$：

$$
W^O = \begin{bmatrix} O_1 \\ O_2 \\ \vdots \\ O_h \end{bmatrix}
$$

根据分块矩阵乘法：

$$
Y = [H_1\;H_2\;\cdots\;H_h]
\begin{bmatrix} O_1 \\ O_2 \\ \vdots \\ O_h \end{bmatrix}
= \sum_{r=1}^{h}H_rO_r
$$

所以，拼接后再投影，可以理解为先把每个头的结果线性变换到输出空间，再把它们相加。这不是简单地对各个头取平均。

## 六、加入 mask 后，公式怎么变化

如果不允许位置 $i$ 读取位置 $j$，可以在 softmax 之前给对应分数加上 $-\infty$。定义加性掩码：

$$
M_{ij} =
\begin{cases}
0, & \text{允许读取位置 }j \\
-\infty, & \text{不允许读取位置 }j
\end{cases}
$$

每个头的公式变为：

$$
H_r = \operatorname{softmax}_{\text{row}}
\left(\frac{Q_rK_r^{\top}}{\sqrt{d_k}}+M\right)V_r
$$

由于 $\exp(-\infty)=0$，被屏蔽位置的权重为零。这里要求每个参与计算的查询行至少有一个允许读取的位置，否则这一行的 softmax 没有正常定义。

对于自回归模型的因果注意力，只允许当前位置读取自己和前面的位置：

$$
M_{ij} =
\begin{cases}
0, & j\leq i \\
-\infty, & j>i
\end{cases}
$$

mask 必须参与 softmax 的归一化过程。如果只在 softmax 之后把部分权重置零，而不重新归一化，剩余权重的和通常就不再是 $1$，与这里的计算不等价。

## 七、用具体维度检查一遍

假设：

$$
n=10,\qquad d_{\text{model}}=512,\qquad h=8,
\qquad d_k=d_v=64
$$

各个阶段的维度如下：

| 对象                                      | 维度             | 含义                   |
| ----------------------------------------- | ---------------- | ---------------------- |
| $X$                                     | $10\times512$  | 10 个 token 的输入表示 |
| $W_r^Q,W_r^K,W_r^V$                     | $512\times64$  | 第$r$ 个头的投影参数 |
| $Q_r,K_r,V_r$                           | $10\times64$   | 单个头中的查询、键和值 |
| $Q_rK_r^{\top}$                         | $10\times10$   | token 两两之间的分数   |
| $A_r$                                   | $10\times10$   | 按行归一化的权重       |
| $H_r=A_rV_r$                            | $10\times64$   | 单个头的输出           |
| $\operatorname{Concat}(H_1,\ldots,H_8)$ | $10\times512$  | 沿特征维度拼接的结果   |
| $W^O$                                   | $512\times512$ | 输出投影参数           |
| $Y$                                     | $10\times512$  | 多头注意力输出         |

常见配置会取 $d_k=d_v=d_{\text{model}}/h$，此时需要 $d_{\text{model}}$ 能被 $h$ 整除。上面的通用公式本身并不要求这个等式，只要求各次矩阵乘法的维度匹配。

实现中还可以把各个头的查询投影横向拼成一个大矩阵：

$$
W^Q = [W_1^Q\;\cdots\;W_h^Q]
\in\mathbb{R}^{d_{\text{model}}\times hd_k}
$$

于是：

$$
XW^Q = [XW_1^Q\;\cdots\;XW_h^Q]
= [Q_1\;\cdots\;Q_h]
$$

这说明“一次投影，再 reshape 成多个头”和“每个头分别投影”在数学上可以等价。加入 batch 维度 $B$ 后，常见的计算布局为 $Q,K\in\mathbb{R}^{B\times h\times n\times d_k}$，softmax 沿最后的 key 位置维度进行。

## 八、自注意力与交叉注意力的区别

前面推导的是自注意力，三组投影都来自同一个输入 $X$。

交叉注意力只需要改变输入来源。设查询序列为 $X_q\in\mathbb{R}^{n_q\times d_{\text{model}}}$，提供上下文的序列为 $X_c\in\mathbb{R}^{n_k\times d_{\text{model}}}$，则：

$$
Q_r=X_qW_r^Q,\qquad K_r=X_cW_r^K,\qquad V_r=X_cW_r^V
$$

此时权重矩阵的维度变为 $n_q\times n_k$，最终输出的维度是 $n_q\times d_{\text{model}}$。输出长度由查询序列决定，key 和 value 则必须在被读取的位置上一一对应。

到这里，得到的是注意力模块本身的输出。Transformer 层中的残差连接、LayerNorm 和前馈网络还需要另行计算。

## 参考

缩放点积注意力与多头结构的定义参见 [Attention Is All You Need，第 3.2 节](https://arxiv.org/html/1706.03762v7#S3.SS2)。本文在此基础上展开逐项计算、分块矩阵乘法和维度检查。
