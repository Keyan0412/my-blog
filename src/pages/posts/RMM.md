---
layout: ../../layouts/PostLayout.astro
title: 论文解析：Reflective Memory Management (RMM)
description: 解析RMM所面向的问题及其解决方案
date: 2026-4-11
---

今天来解析一篇针对 **多会话对话智能体** 进行记忆构建的文章。

原文标题： **In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents**

-----
## 1. 问题解析与构建
首先，文中对问题的场景进行了详细的设定：该论文针对的是 **多会话场景** ，而会话边界则是由用户的闲置、显式的用户分段请求、以及用户开启新的会话线程所区分。个人认为，这种会话可以理解为使用类ChatGPT应用时，在页面左侧会话栏中显示的会话。
基于此背景，文中指出当前构建智能体记忆能力的方法的两个主要问题。首先， **固定的记忆粒度** 会破坏自然语义的结构，导致信息碎片化问题。同时， **固定的检索机制** 无法适应多变的对话内容与用户交互模式。
因此，要构建能长期稳定的个性化对话智能体，不仅要识别并储存重要的用户信息以备未来检索的需求，同时需要设计检索方案从而准确检索过往储存的信息，从而防止检索失准导致的注意力偏移问题。

-----
## 2. 解决方案
针对该问题，文中提出一种名为 **Reflective Memory Management（RMM）** 的方案。该方案包含两种机制： **Prospective Reflection** 与 **Retrospective Reflection** 。其中， Prospective Reflection 负责解构对话历史，并生成基于话题的记忆表现。具体而言，处理后的对话历史由多个形似（话题总结，原对话）的元组组成。而 Retrospective Reflection 负责通过在线反馈来动态调整检索机制。

该框架包含四个关键组件：
 - **Memory bank**：用于储存先前生成的记忆。
 - **Retriever**：从 Memory bank 中检索相关记忆。
 - **Reranker**：对 Retriever 的结果进行动态调整。
 - **LLM**：结合 Reranker 的结果生成回答，同时在 Reranker 的训练中充当在线反馈的角色。

以上工作流可由下图辅助理解：
![workflow](/my-blog//images/RMM/workflow.png)

### 2.1. Prospective Reflection
如前文所说，Prospective Reflection 负责将对话转换为基于话题的记忆。该转化过程包含两个步骤，即 **记忆提取（Memory extraction）** 与 **记忆更新（Memory update）**。在记忆提取阶段，作者使用大模型将完整对话转化为了对话片段以及话题总结，从而构建前文提到的记忆元组；在记忆更新阶段，作者根据新生成的记忆对 Memory bank 进行检索，若有极为相似的则进行合并，否则增设新的记忆条目。下入中展示了 Prospective Reflection 的工作流示意：
![Prospective Reflection](/my-blog//images/RMM/prospective.png)
**个人理解**： 该过程将冗杂的、充满噪音的原文本转换为了语义信息更为清晰的话题总结，从而利好于之后的RAG检索。

### 2.2. Retrospective Reflection
在进行检索时，使用固定的检索器（此处特质密集检索器，或通常理解为编码器）往往无法很好地适应多样化的文本。而根据对话内容对检索器进行微调不仅需要大量计算资源，同时还容易导致灾难性遗忘。因此，本文中使用了一种轻量化的重排序器（Reranker），用于从检索器获取 Top-K 结果并重新输出更相关的 Top-M 结果，并基于大模型对记忆的使用情况对自身进行更新。
![Retrospective Reflection](/my-blog//images/RMM/retrospective.png)
在通过检索器获取编码及 Top-K 后，作者使用了代残差的线性层对原 query 以及 memory 的编码进行线性转化。假设 query 的编码为 $q$，第 $i$ 条 memory 的编码为 $m_i$，那么计算过程为
$$
q^{,} = q + W_qq, m^,_i = m_i + W_mm_i
$$
之后，通过 $s_i = q^{,T}m^,_i$ 即可获得相关性分数。然而，如果单纯使用当前分数选出 Top-M，则有用但分数低的记忆被选中的概率较低，其分数也难以通过训练进行校正。因此，作者引入了 **Gumbel Trick**，通过给分数加上 Gumbel 噪声:
$$
\widetilde{s}_i = s_i+g_i, g_i=-log(-log(u_i)), u_i\sim\text{Uniform}(0, 1)
$$
再用 softmax 得到采样概率：
$$
p_i = \frac{exp(\widetilde{s}_i/\tau)}{\sum_{j=1}^Kexp(\widetilde{s}_j/\tau)}
$$
并通过该采样概率获得 Top-M（此处 $\tau$ 为温度参数。温度越低，则分数越高的记忆越容易选中，检索结果也更具有确定性）。之后，大模型将基于 Top-M 和 query 生成回复，并在回复的过程中同步生成 citation ，用于决定 Reranker 在参数更新时的 Return。在记忆被用到时，Reward 为 +1，否则为 -1。之后，Reranker的参数将按照以下式子进行更新：
$$
\Delta\theta = \alpha(R - b)\Delta_\theta log P(M_M|q, M_K;\theta)
$$
其中，$\theta$ 为 Reranker 参数，$\alpha$ 为学习率，$R$ 和 $b$ 分别为单步 Return 和 Baseline。该过程的本质是对 Reranker 进行单步 REINFORCE 训练。
