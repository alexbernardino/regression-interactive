# Signal — Linear Regression Laboratory

An interactive teaching tool for exploring two-dimensional linear regression,
noise, outliers, L1/L2 regularization, evaluation metrics, and parameter
uncertainty.

The experiment runs entirely in the browser. It does not require a database,
API server, GPU, or virtual machine.

## Develop locally

Requirements: Node.js 22 or newer.

```bash
npm install
npm run dev
```

Open the local address displayed in the terminal. Changes to the source appear
automatically while the development server is running.

## Verify the GitHub Pages build

```bash
npm run build:pages
```

The deployable static site is generated in `out/`.

To simulate a project repository named `regression-demo`, including its URL
prefix, run:

```bash
PAGES_BASE_PATH=/regression-demo npm run build:pages
```

## Publish with GitHub Pages

1. Create a GitHub repository and upload this project to its `main` branch.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, choose **GitHub Actions** as the source.
4. Open the **Actions** tab and wait for **Deploy to GitHub Pages** to finish.

Every later push to `main` rebuilds and republishes the site automatically.
The workflow reads GitHub's own Pages base path, so it works both at
`https://USERNAME.github.io/REPOSITORY/` and with a configured custom domain.

## Available commands

- `npm run dev` — start the local interactive preview
- `npm run build` — build the existing vinext/Cloudflare version
- `npm run build:pages` — create the GitHub Pages static export
- `npm run lint` — check the source code
