import type { NextConfig } from "next";

/**
 * Next.js Configuration — Anchorium Omni-Engine
 *
 * Key customizations:
 *   - Turbopack: Configured with resolve aliases to exclude Node-only
 *     ONNX modules from the browser bundle (Transformers.js compatibility).
 *   - Headers: COOP/COEP for SharedArrayBuffer support (Transformers.js perf).
 */
const nextConfig: NextConfig = {
  // Turbopack config (Next.js 16+ default bundler)
  turbopack: {
    resolveAlias: {
      // Transformers.js uses onnxruntime-node on the server side,
      // but we only need onnxruntime-web in the browser.
      "onnxruntime-node": { browser: "" },
      "sharp": { browser: "" },
    },
  },

  // Fallback webpack config (for `next build --webpack`)
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-node": false,
        "sharp": false,
      };
    }
    return config;
  },

  // Allow loading pdf.js from CDN and enable SharedArrayBuffer
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "credentialless",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
