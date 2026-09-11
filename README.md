# 可言的博客

## 博客分类与发布

实际发布的文章位于 `src/pages/posts/<分类>/`，博客首页 `/blog/` 显示分类文件夹，
`/blog/<分类>/` 显示该分类的文章。以下路径均相对于站点 base `/my-blog`。

| 文件夹 | 页面分类 | 内容 |
| --- | --- | --- |
| `agents` | Agent 设计 | Agent 演进、Skills、仓库架构与上下文策略 |
| `inference` | 推理系统 | Mini-SGLang 与推理框架解析 |
| `papers-algorithms` | 论文与算法 | RMM 论文解析、REINFORCE 推导 |
| `engineering` | 工程与基础设施 | App Infra 系列、SDK 开发 |
| `notes` | 随笔 | 欢迎文章与个人记录 |

新增文章时，将 Markdown 文件放入对应文件夹，并使用以下 frontmatter：

```yaml
---
layout: ../../../layouts/PostLayout.astro
category: agents
title: 文章标题
description: 文章简介
date: 2026-9-11
---
```

`category` 必须与文件夹名称一致。分类页会自动收录文章，Markdown 标题会自动生成可折叠大纲。
新增分类时，在 `src/lib/categories.ts` 中添加名称、slug 和简介，再创建同名文章文件夹。
文章地址为 `/posts/<分类>/<文件名>/`；`src/pages/posts/[slug].astro` 为旧地址
`/posts/<文件名>/` 生成静态跳转页，因此不同分类中的文章文件名应保持唯一。

`src/content/blog/` 是初始模板的示例集合，不属于当前博客分类列表。

## 原始模板说明

```sh
npm create astro@latest -- --template blog
```

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

Features:

- ✅ Minimal styling (make it your own!)
- ✅ 100/100 Lighthouse performance
- ✅ SEO-friendly with canonical URLs and Open Graph data
- ✅ Sitemap support
- ✅ RSS Feed support
- ✅ Markdown & MDX support

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   ├── content/
│   ├── layouts/
│   └── pages/
├── astro.config.mjs
├── README.md
├── package.json
└── tsconfig.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

The `src/content/` directory contains "collections" of related Markdown and MDX documents. Use `getCollection()` to retrieve posts from `src/content/blog/`, and type-check your frontmatter using an optional schema. See [Astro's Content Collections docs](https://docs.astro.build/en/guides/content-collections/) to learn more.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## 👀 Want to learn more?

Check out [our documentation](https://docs.astro.build) or jump into our [Discord server](https://astro.build/chat).

## Credit

This theme is based off of the lovely [Bear Blog](https://github.com/HermanMartinus/bearblog/).
