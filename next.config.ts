import type { NextConfig } from "next";

const isGitHubPagesBuild = process.env.GITHUB_PAGES === "true";
const pagesBasePath = process.env.PAGES_BASE_PATH ?? "";

if (pagesBasePath && !pagesBasePath.startsWith("/")) {
  throw new Error("PAGES_BASE_PATH must be empty or start with a slash.");
}

const nextConfig: NextConfig = isGitHubPagesBuild
  ? {
      // GitHub Pages only serves static files. The workflow publishes `out/`.
      output: "export",
      basePath: pagesBasePath,
      trailingSlash: true,
      images: {
        unoptimized: true,
      },
      typescript: {
        // The Pages build has no Cloudflare worker or database runtime.
        tsconfigPath: "tsconfig.pages.json",
      },
    }
  : {};

export default nextConfig;
