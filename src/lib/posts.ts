import type { MarkdownInstance } from "astro";
import { categories } from "./categories";

interface Frontmatter {
  title: string;
  description?: string;
  category: string;
}

export const posts = Object.values(
  import.meta.glob<MarkdownInstance<Frontmatter>>("../pages/posts/*/*.md", { eager: true }),
);

for (const post of posts) {
  const folder = post.file.split("/").at(-2);
  if (!categories.some(({ slug }) => slug === folder) || post.frontmatter.category !== folder) {
    throw new Error(`Invalid post category: ${post.file}`);
  }
}

export const postsInCategory = (slug: string) =>
  posts.filter((post) => post.frontmatter.category === slug);
