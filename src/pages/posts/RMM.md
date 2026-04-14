---
layout: ../../layouts/PostLayout.astro
title: 论文解析：Reflective Memory Management (RMM)
description: 解析RMM所面向的问题及其解决方案
date: 2026-4-11
---

今天来解析一篇面向 **多会话对话智能体** 记忆构建的文章。

原文标题： **In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents**

-----
## 1. 问题解析与构建
首先，论文对问题场景做了较为细致的设定：其目标场景是 **多会话场景** ，而会话边界由用户闲置、显式的分段请求，以及用户开启新的会话线程来划分。个人理解上，这里的“会话”可以类比为 ChatGPT 一类产品左侧会话栏中的单个会话线程。

在这一背景下，作者指出当前智能体记忆构建方法存在两个主要问题。首先， **固定的记忆粒度** 会破坏自然语义结构，带来信息碎片化；其次， **固定的检索机制** 难以适应不断变化的对话内容与用户交互模式。

因此，若想构建长期稳定的个性化对话智能体，不仅需要识别并存储重要的用户信息，以便未来检索，还需要设计合适的检索方案，使系统能够准确找回过往信息，从而避免因检索失准而产生注意力偏移。

-----
## 2. 解决方案
针对上述问题，作者提出了一种名为 **Reflective Memory Management（RMM）** 的方案。该方案包含两种机制： **Prospective Reflection** 与 **Retrospective Reflection** 。其中，Prospective Reflection 负责解构对话历史，并生成基于话题的记忆表示。具体来说，处理后的对话历史会被组织为多个形如“话题总结-原对话”的元组；而 Retrospective Reflection 则通过在线反馈动态调整检索机制。

该框架包含四个关键组件：
- **Memory bank**：用于存储先前生成的记忆。
- **Retriever**：从 Memory bank 中检索相关记忆。
- **Reranker**：对 Retriever 的结果进行动态调整。
- **LLM**：结合 Reranker 的结果生成回答，同时在 Reranker 的训练中充当在线反馈的来源。

以上工作流可由下图辅助理解：
![workflow](/my-blog//images/RMM/workflow.png)

### 2.1. Prospective Reflection
如前文所述，Prospective Reflection 负责将对话转换为基于话题的记忆。该过程包含两个步骤，即 **记忆提取（Memory extraction）** 与 **记忆更新（Memory update）**。在记忆提取阶段，作者使用大模型将完整对话转化为对话片段及其话题总结，从而构建前文提到的记忆元组；在记忆更新阶段，作者会基于新生成的记忆检索 Memory bank，若发现高度相似的已有记忆则进行合并，否则新增一条记忆。下图展示了 Prospective Reflection 的工作流示意：
![Prospective Reflection](/my-blog//images/RMM/prospective.png)
**个人理解**：该过程实际上是将冗杂、噪声较多的原始对话压缩为语义更清晰的话题总结，因此会更利于后续的 RAG 检索。

### 2.2. Retrospective Reflection
在检索阶段，使用固定的检索器（这里本质上是密集检索器，也可以简单理解为编码器）往往难以很好适应多样化文本。而如果根据对话内容持续对检索器做微调，一方面计算开销较大，另一方面也容易引发灾难性遗忘。因此，本文引入了一种轻量化的重排序器（Reranker），用于对检索器返回的 Top-K 结果重新排序，输出更相关的 Top-M 结果，并基于大模型对记忆的实际使用情况对自身进行更新。
![Retrospective Reflection](/my-blog//images/RMM/retrospective.png)
在通过检索器获得编码及 Top-K 结果后，作者使用带残差的线性层对原始 query 与 memory 的编码进行线性变换。假设 query 的编码为 $q$，第 $i$ 条 memory 的编码为 $m_i$，则计算过程为
$$
q^{,} = q + W_qq, m^,_i = m_i + W_mm_i
$$
之后，通过 $s_i = q^{,T}m^,_i$ 即可得到相关性分数。然而，如果仅依据当前分数直接选出 Top-M，那么那些“有用但当前分数偏低”的记忆被采样到的概率会很小，它们的分数也就难以在训练过程中得到有效校正。因此，作者引入了 **Gumbel Trick**，通过给分数加入 Gumbel 噪声：
$$
\widetilde{s}_i = s_i+g_i, g_i=-log(-log(u_i)), u_i\sim\text{Uniform}(0, 1)
$$
再通过 softmax 得到采样概率：
$$
p_i = \frac{exp(\widetilde{s}_i/\tau)}{\sum_{j=1}^Kexp(\widetilde{s}_j/\tau)}
$$
并基于该采样概率选出 Top-M（这里 $\tau$ 为温度参数。温度越低，分数越高的记忆越容易被选中，检索结果也越具有确定性）。随后，大模型会基于 Top-M 和 query 生成回复，并在回复过程中同步生成 citation，用于决定 Reranker 参数更新时的 Return。若某条记忆被实际用到，则 Reward 记为 +1，否则记为 -1。之后，Reranker 的参数按下式更新：
$$
\Delta\theta = \alpha(R - b)\Delta_\theta log P(M_M|q, M_K;\theta)
$$
其中，$\theta$ 为 Reranker 参数，$\alpha$ 为学习率，$R$ 和 $b$ 分别表示单步 Return 与 Baseline。该过程的本质，是对 Reranker 进行单步 REINFORCE 训练。
