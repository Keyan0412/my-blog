import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  site: "https://keyan0412.github.io",
  base: "/my-blog",
  trailingSlash: "always",
  vite: {
    plugins: [tailwindcss()],
  },
});