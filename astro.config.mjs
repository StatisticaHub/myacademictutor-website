// @ts-check
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://statisticahub.github.io",
  base: "/myacademictutor-website",
  integrations: [sitemap()],
  vite: {
    plugins: [tailwindcss()],
  },
});