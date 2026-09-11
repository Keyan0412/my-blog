export const categories = [
  { slug: "agents", title: "Agent 设计", description: "从演进路线到 Skills、仓库架构与上下文策略。" },
  { slug: "inference", title: "推理系统", description: "深入 Mini-SGLang 等大模型推理框架的实现。" },
  { slug: "papers-algorithms", title: "论文与算法", description: "阅读记忆管理论文，推导强化学习算法。" },
  { slug: "engineering", title: "工程与基础设施", description: "App Infra 系列，以及 SDK 与产品工程实践。" },
  { slug: "notes", title: "随笔", description: "博客的开始，以及学习之外的记录。" },
] as const;

export const categoryUrl = (slug: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, "")}/blog/${slug}/`;
