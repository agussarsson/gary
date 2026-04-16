import type { NextConfig } from "next";

/** Server-side proxy target (browser uses same-origin /backend-api/... to avoid cross-origin fetch issues). */
const apiProxyTarget =
  process.env.GARY_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  turbopack: { root: __dirname },
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [
      {
        source: "/backend-api/:path*",
        destination: `${apiProxyTarget.replace(/\/$/, "")}/:path*`,
      },
    ];
  },
};

export default nextConfig;
