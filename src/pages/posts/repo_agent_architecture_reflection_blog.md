---
layout: ../../layouts/PostLayout.astro
title: 构建仓库 Agent 的反思
description: 探究多 Agent 分工的合理模式
date: 2026-5-1
---

# 从 FileReader 到 Investigator：一次 Repo Agent 架构重构的反思

最近在设计一个 repo-agent 的过程中，我遇到了一个很典型的问题：当一个 Agent 需要理解代码仓库时，它到底应该如何分工，才能既保持上下文干净，又不损失对整体结构的理解能力？

最开始，我的设计比较直接。系统中有一个 MainAgent，负责和用户交互、维护 EvidenceGraph、生成最终回答；当 MainAgent 发现证据不足时，它会请求 InvestigatorAgent 去调查；而 InvestigatorAgent 为了避免自己直接读取太多文件，又会调用 FileReaderAgent，让 FileReaderAgent 针对某个文件给出回答。

这个设计初看是合理的。MainAgent 不碰代码细节，InvestigatorAgent 负责调查，FileReaderAgent 负责读取文件。三层结构看起来职责清晰，也符合我一开始对多 Agent 系统的理解：把不同能力拆开，让每个 Agent 只处理自己擅长的事情。

但在继续推演后，我发现这里面有一个很隐蔽的问题：**FileReaderAgent 的抽象层级太低了。**

它只能看到一个文件，因此它回答问题的能力天然受到当前文件视野的限制。如果 InvestigatorAgent 已经知道该问什么、该读哪个文件，那么 FileReaderAgent 确实能很好地发挥作用。问题是，真正困难的场景往往不是“我知道要读哪个文件”，而是“我还不知道应该读哪些文件”。

换句话说，原来的设计解决了上下文爆炸，但没有很好地解决冷启动和跨文件理解的问题。

---

## 一开始的困境：上下文爆炸与局部视野

在原架构中，InvestigatorAgent 的职责其实比较重。它需要理解 MainAgent 发来的调查任务，需要在 repo 中搜索相关文件，需要判断哪些文件值得阅读，还需要把 FileReaderAgent 返回的结果综合成 InvestigationReport。

于是问题出现了。

如果 InvestigatorAgent 自己读文件，它的上下文很快会爆炸。一个稍微复杂一点的项目，相关文件可能有十几个，函数调用链可能跨越多个模块。让 InvestigatorAgent 直接把这些内容都塞进自己的上下文，显然不是一个长期可维护的方案。

于是我引入 FileReaderAgent，希望通过“文件级问答”的方式降低上下文压力。InvestigatorAgent 不直接读文件，而是向 FileReaderAgent 提问，例如：

```text
请阅读 src/tools/trace_symbol.py，判断 trace_symbol 是如何定位符号定义的。
```

FileReaderAgent 只读取当前文件，然后返回带行号的回答。这样确实能减少 InvestigatorAgent 的上下文负担。

但很快我又发现另一个问题：FileReaderAgent 的回答质量高度依赖 InvestigatorAgent 的提问质量。问题问得具体，它就回答得具体；问题问得模糊，它就只能泛泛总结。如果 InvestigatorAgent 在冷启动阶段根本不知道 repo 的结构，也不知道关键符号在哪里，那么它问出来的问题很可能就是低质量的。

这就产生了一种尴尬局面：

```text
InvestigatorAgent 需要符号知识，才能提出高质量问题；
但它又需要通过 FileReaderAgent，才能获得符号知识。
```

这其实是一个冷启动悖论。

---

## 后来我意识到：问题不在于是否需要 FileReader，而在于分层错了

继续思考之后，我逐渐意识到，原来的分层其实有一点别扭。

原架构大致是：

```text
MainAgent
  └── InvestigatorAgent
        └── FileReaderAgent
```

其中，InvestigatorAgent 夹在中间，既要负责问题拆解，又要负责调查路径规划，还要负责组织 FileReaderAgent 的提问。它的职责并不纯粹。

而 FileReaderAgent 的粒度又太小。真实的软件理解并不是按文件完成的，而是按问题完成的。

比如用户问：

```text
这个项目是如何体现 ReAct 架构的？
```

这显然不是一个单文件问题。它可能涉及 MainAgent 的控制流、工具调用机制、消息历史维护、InvestigationReport 的生成、EvidenceGraph 的使用方式等。让一个 FileReaderAgent 去读某个文件，最多只能提供局部信息。真正有价值的调查单元应该是：

```text
项目中是否存在循环式的 reasoning/action/observation 控制流？
工具调用结果如何回到模型上下文？
MainAgent 和 InvestigatorAgent 的职责边界在哪里？
EvidenceGraph 如何参与最终回答？
```

这些都是“问题级”的调查，而不是“文件级”的阅读。

所以我开始重新考虑：是不是应该把上层的 InvestigatorAgent 改造成 AnalyzerAgent，而把下层的 FileReaderAgent 升级成真正的 InvestigatorAgent？

---

## 新的分层：Main 不变，下面重新洗牌

新的结构变成：

```text
MainAgent
  └── AnalyzerAgent
        └── InvestigatorAgent
```

这不是简单改名，而是职责重排。

MainAgent 仍然保持不变。它只负责 EvidenceGraph、InvestigationReport、Claim 和 Final Answer。它不直接碰 repo，不直接读文件，也不直接进行底层调查。这个边界我认为必须保留，因为 MainAgent 是整个系统的高层推理中心，一旦它开始处理文件细节，EvidenceGraph-driven 的设计就会被破坏。

AnalyzerAgent 是原来的 InvestigatorAgent 改造而来。它不再亲自深入调查 repo，而是负责分析 MainAgent 发来的大问题，并将其拆成若干可调查的小问题。

InvestigatorAgent 则是原来的 FileReaderAgent 升级而来。它不再只是“读取单个文件并回答”，而是围绕一个明确子问题，在 repo 中搜索、定位、读取多个必要文件，并给出带证据的子调查报告。

也就是说，新结构中的职责变成：

```text
MainAgent：判断还缺什么证据，以及如何基于证据生成最终回答。

AnalyzerAgent：判断一个大调查任务应该拆成哪些子问题。

InvestigatorAgent：判断每个子问题需要读哪些文件，并提取证据回答。
```

这个分层比原来更加自然。

---

## 为什么 Analyzer 不应该读文件

这里有一个很重要的边界：AnalyzerAgent 不应该直接读取完整文件。

原因很简单。如果 AnalyzerAgent 可以随意读文件，那么它很快又会退化成原来的 InvestigatorAgent。它会开始搜索文件、理解文件、总结证据，最后上下文又会变重。

AnalyzerAgent 的价值不是掌握细节，而是掌握问题结构。

它应该接收这样的任务：

```text
调查这个项目如何体现 ReAct 架构。
```

然后拆成这样的子问题：

```text
项目中是否存在循环式的 agent 控制流？

工具调用是如何被选择、执行和回传的？

工具结果是否会进入下一轮模型上下文？

EvidenceGraph 是否参与主 Agent 的最终推理？
```

这些子问题应该是 question-centered，而不是 file-centered。也就是说，AnalyzerAgent 不应该输出：

```text
读取 main_agent.py。
读取 investigator_agent.py。
读取 tools.py。
```

这种拆分看似具体，实际上很容易误导下层 Agent。因为 AnalyzerAgent 还没有真正调查，它并不知道哪些文件最关键。它应该把“读哪些文件”的决策交给 InvestigatorAgent。

AnalyzerAgent 真正要做的是定义调查方向，而不是指定调查材料。

---

## Investigator 的粒度应该是子问题，而不是文件

新架构中的 InvestigatorAgent 和原来的 FileReaderAgent 最大区别在于：它的基本任务单位从“文件”变成了“子问题”。

原来的 FileReaderAgent 输入大概是：

```text
file_path + question
```

新的 InvestigatorAgent 输入应该是：

```text
subquestion + purpose + expected_evidence + search_hints + budget
```

例如：

```text
子问题：工具调用结果如何回到模型上下文？

目的：判断项目是否具有 ReAct 风格中的 observation feedback。

期望证据：
- tool result 被包装成 message
- message 被 append 回 history
- 下一轮模型调用能够看到该结果

搜索线索：
tool_call, observation, messages, history, append
```

InvestigatorAgent 收到这个任务后，可以自己决定先用 find_text 搜索关键词，再用 trace_symbol 定位函数，再读取相关文件片段。它不需要理解完整的用户大问题，只需要把当前子问题查清楚。

这就避免了两个极端：

```text
MainAgent 太重，直接陷入代码细节。

FileReaderAgent 太轻，只能看到单个文件。
```

新的 InvestigatorAgent 位于一个更合适的位置：它有明确的问题目标，也有足够的文件访问能力。

---

## MainAgent 为什么应该保持不变

在这次重构里，我最不想动的是 MainAgent。

原来的 MainAgent 设计其实是比较正确的。它只处理 EvidenceGraph，不处理 repo 细节；证据不足时调用 request_investigation；调查完成后通过 derive_claim 把有价值结论写入 EvidenceGraph；最终回答必须基于 evidence id。

这套机制的重点是：MainAgent 面对的是“被压缩后的证据世界”，而不是“完整的调查过程”。

如果把 AnalyzerAgent 和 InvestigatorAgent 的中间过程全部暴露给 MainAgent，MainAgent 的上下文还是会膨胀。更糟糕的是，它会在大量未正式采纳的 observation 中迷失，从而破坏 EvidenceGraph 的价值。

因此，我希望 MainAgent 完全不关心下层怎么调查。它只知道：

```text
我请求了一次 investigation。
我拿到了一个 InvestigationReport。
我可以基于这个 report derive claim。
```

至于这个 InvestigationReport 是由旧的 InvestigatorAgent 直接生成，还是由新的 AnalyzerAgent 拆解后汇总生成，对 MainAgent 来说应该是透明的。

这也是我倾向于让 AnalyzerAgent 对外实现旧接口的原因：

```python
class AnalyzerAgent:
    def investigate(self, task: InvestigationTask) -> InvestigationReport:
        ...
```

这样 MainAgent 甚至不需要知道自己下面换了一个 Agent。

---

## 新架构中的数据流

重构后的运行过程大致如下：

```text
User Query
  ↓
MainAgent
  ↓
request_investigation
  ↓
AnalyzerAgent
  ↓
AnalysisPlan
  ↓
SubInvestigationTask[]
  ↓
InvestigatorAgent
  ↓
SubInvestigationReport[]
  ↓
AnalyzerAgent synthesis
  ↓
InvestigationReport
  ↓
MainAgent
  ↓
derive_claim / final_answer
```

这条链路中，每一层处理的信息粒度都不同。

MainAgent 处理 evidence。

AnalyzerAgent 处理 question decomposition。

InvestigatorAgent 处理 code evidence。

这个分层的好处是，每个 Agent 都只需要保留与自己职责相关的上下文。MainAgent 不需要知道文件内容，AnalyzerAgent 不需要掌握完整代码细节，InvestigatorAgent 不需要承担完整用户问题的综合解释。

这才是多 Agent 系统真正有意义的地方。

---

## 我会如何调整目录结构

如果按照这个思路改造，原来的目录结构也应该随之变化。

原本的：

```text
agents/
  main_agent.py
  investigator_agent.py
  file_reader_agent.py
```

可以调整为：

```text
agents/
  main_agent.py
  analyzer_agent.py
  investigator_agent.py
```

同时，`investigation/` 目录里应该增加与分解任务相关的数据结构：

```text
investigation/
  task.py
  plan.py
  subtask.py
  report.py
  subreport.py
  observation.py
  scratchpad.py
```

其中：

```text
task.py
  保存 MainAgent 发给 AnalyzerAgent 的调查任务。

plan.py
  保存 AnalyzerAgent 生成的 AnalysisPlan。

subtask.py
  保存 AnalyzerAgent 发给 InvestigatorAgent 的子调查任务。

subreport.py
  保存 InvestigatorAgent 返回的子调查报告。

report.py
  保存最终交给 MainAgent 的 InvestigationReport。
```

工具权限也应该重新划分。

MainAgent 仍然只能使用：

```text
request_investigation
derive_claim
final_answer
```

AnalyzerAgent 可以使用：

```text
request_subinvestigation
```

如果需要冷启动能力，也可以给 AnalyzerAgent 很轻量的 `read_repo_tree` 或 `find_text`，但我倾向于谨慎开放。AnalyzerAgent 一旦拥有过强的 repo 操作能力，就很容易重新变成重型调查 Agent。

InvestigatorAgent 则可以使用：

```text
read_repo_tree
find_text
trace_symbol
read_file
```

因为它才是真正负责子问题调查的 Agent。

---

## 一个关键变化：删除 ask_file

在原架构里，InvestigatorAgent 不能直接读文件，只能通过 `ask_file` 调用 FileReaderAgent。这是为了防止它把文件内容全部塞进上下文。

但在新架构中，InvestigatorAgent 的任务已经被 AnalyzerAgent 限定成一个小问题，同时又有 `max_tool_calls` 和 `max_files` 的预算限制。因此，它可以直接使用 `read_file`。

这时继续保留 `ask_file` 反而会显得多余。

原来的链路是：

```text
InvestigatorAgent
  ↓
ask_file
  ↓
FileReaderAgent
  ↓
read_file
```

新的链路可以直接变成：

```text
InvestigatorAgent
  ↓
read_file
```

这并不是放松约束，而是把约束从“不能读文件”改成了“只能围绕一个子问题，在预算内读必要文件”。

这个约束更符合实际。

---

## 这次重构真正改变了什么

表面上看，这次只是把原来的 InvestigatorAgent 改成 AnalyzerAgent，把 FileReaderAgent 改成 InvestigatorAgent。

但我认为真正改变的是系统的抽象单位。

原来的抽象单位是：

```text
文件
```

现在的抽象单位是：

```text
问题
```

这点非常重要。

代码仓库不是一本按顺序读完的书，而是一张由模块、符号、调用关系、配置和测试共同组成的网络。理解一个 repo 的过程，本质上不是“读完所有文件”，而是“围绕问题找到足够证据”。

所以，一个 repo-agent 不应该以文件为核心组织推理，而应该以问题为核心组织调查。

FileReaderAgent 当然有价值，但它更适合作为一种能力，而不是一个独立的核心 Agent。真正值得成为 Agent 的，是能围绕子问题主动搜索、读取、比较、判断证据的 Investigator。

---

## 我目前形成的设计原则

经过这次思考，我会把新的架构原则总结成几句话：

```text
MainAgent 只处理 evidence，不处理 repo 细节。

AnalyzerAgent 只处理问题分解和调查综合，不亲自做深度文件阅读。

InvestigatorAgent 只处理单个子问题的证据调查，不回答完整用户问题。

EvidenceGraph 只保存被 MainAgent 采纳的 claim。

request_investigation 只能被 MainAgent 调用。

request_subinvestigation 只能被 AnalyzerAgent 调用。

read_file 只能被 InvestigatorAgent 调用。

derive_claim 只能被 MainAgent 调用。
```

这组原则背后的核心其实很简单：

```text
高层负责判断意义。
中层负责拆分问题。
底层负责寻找证据。
```

只要这三个层级不混在一起，系统就比较不容易失控。

---

## 我对这次重构的判断

如果说原来的设计解决的是“如何防止 MainAgent 看到太多文件内容”，那么新的设计进一步解决的是“如何防止 InvestigatorAgent 在一无所知时乱读文件”。

原来的 FileReaderAgent 确实能缓解上下文爆炸，但它无法解决问题拆解。新的 AnalyzerAgent 则专门负责这个问题。

从这个角度看，新架构并不是把系统做复杂了，而是把复杂性放到了更合适的位置。

之前的复杂性堆在 InvestigatorAgent 身上：

```text
搜索、读文件、提问、理解、综合，全都在一个 Agent 里。
```

现在复杂性被拆开：

```text
AnalyzerAgent 负责拆问题。
InvestigatorAgent 负责查证据。
MainAgent 负责采纳证据并回答。
```

我认为这更接近一个真正可维护的 repo analysis agent。

---

## 结语

这次重构给我的一个启发是：多 Agent 系统的关键不在于“多”，而在于每一层的抽象是否正确。

如果抽象错了，多加一个 Agent 只会多一层沟通成本。原来的 FileReaderAgent 就有这个问题，它虽然隔离了文件内容，却没有很好地承载代码理解中的核心任务。

而当我把上层改成 Analyzer、下层改成 Investigator 后，职责边界反而清晰了很多。

MainAgent 不再关心调查过程，只关心证据。

AnalyzerAgent 不再亲自读代码，只关心问题如何拆。

InvestigatorAgent 不再只读单个文件，而是围绕子问题寻找证据。

这套结构对我来说更符合 repo-agent 的本质：它不是一个会读文件的聊天机器人，而是一个围绕问题组织代码证据的分析系统。

后续真正需要打磨的，应该不是再继续增加 Agent 数量，而是把几个协议设计扎实：

```text
InvestigationTask
AnalysisPlan
SubInvestigationTask
SubInvestigationReport
InvestigationReport
EvidenceGraph
```

这些结构稳定了，Agent 之间的协作才会稳定。否则，无论起多少个 Agent，本质上都只是把混乱从一个上下文搬到另一个上下文而已。
