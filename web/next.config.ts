import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(__dirname, ".."),
  },
  async rewrites() {
    return [
      {
        source: "/api/vtax/:path*",
        destination: `${process.env.VTAX_API_URL ?? "http://127.0.0.1:8100"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
