# Web 前端本地开发（Ubuntu）

完整生产部署请阅读仓库根目录的 [DEPLOYMENT.md](../DEPLOYMENT.md)。本页仅说明 Ubuntu 上的前端开发方式。

## 前置条件

- Node.js 22 LTS
- 已在 `127.0.0.1:8100` 启动 VTAX FastAPI 编排服务

## 启动

在仓库根目录执行：

```bash
npm ci
VTAX_API_URL=http://127.0.0.1:8100 npm run dev:web -- --hostname 0.0.0.0 --port 3001
```

浏览器访问：

```text
http://127.0.0.1:3001
```

局域网访问时，将 `127.0.0.1` 替换为 Ubuntu 服务器 IP。当前版本没有登录鉴权，不要直接暴露到公网。
