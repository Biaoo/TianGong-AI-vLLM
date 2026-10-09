# GPUStack 控制面与模型网关

GPUStack 负责 GPU worker、模型实例生命周期与统一推理入口；模型镜像、固定权重和验收仍由本仓库维护。文档解析 API、Celery、任务队列及结果存储属于 Unstructure 项目。

## 控制面

版本与镜像摘要固定在 `deploy/gpustack/release.json`。Compose 项目为 `tiangong-gpustack`，服务为 `server`，持久卷为 `tiangong-gpustack_data`。控制面禁用内嵌 worker，不分配 GPU、不挂载 Docker socket。私有配置 `deployments/gpustack/server.json` 只包含 host/port，不进入原生模型的 all 范围。

首次安装：

```bash
bash deploy/gpustack/manage.sh init
# 编辑 deployments/gpustack/server.json，默认只绑定回环。
bash deploy/gpustack/manage.sh config
bash deploy/gpustack/manage.sh pull
bash deploy/gpustack/manage.sh start
bash deploy/gpustack/manage.sh status
```

已有部署不要重新 init 或创建空卷覆盖原数据。注册 worker 时控制面必须绑定明确的 LAN IPv4，双方端口可达。Ubuntu Docker 安装辅助脚本 `deploy/gpustack/install-docker-ubuntu.sh` 默认 --check 只读，--install 保留 daemon 配置和转发策略；已有 Docker 不自动覆盖。

用户名 admin，首次凭据保存在权限 600 的 `.secrets/gpustack/bootstrap-password`；UI 修改密码后该文件只是初始化资料。API key 保存到 `.secrets/gpustack/api-key`，只授予所需 inference 路由。不要打印展开后的 Compose、注册 token 或数据库内容。

start 不重建、不拉取、不构建；配置变更使用 restart。stop 保留容器及持久卷。单控制面维护会中断统一网关，多个模型副本不等于控制面高可用。

## 模型与路由

worker 操作、挂载与参数见 [模型托管](gpustack-workers.md)。同一模型实例只能由 GPUStack 或原生部署器之一管理；先排空旧实例再切换，不能同时占用同组 GPU。

| 主路由 | 调用协议 | 实例拓扑 |
| --- | --- | --- |
| qwen38 | /v1，model=qwen38；Generic Proxy 可保留原始模型别名 | 每节点一个 TP4 + EP 实例 |
| nemotron-embed | Generic Proxy 下 /v2/embed，保留 query/document | 每节点一个 DP3 或 DP4 实例 |
| mineru4 | Generic Proxy，MinerU 使用完整 route URL 与专用 API key | 每节点一个 DP3 或 DP4 实例 |

Generic Proxy 前缀为 `/model/proxy/ROUTE_ID`。调用方记录稳定主路由，不能配置 worker 的动态实例端口。节点专用路由用于诊断。更新 route 时须传目标对象列表；GET 响应的 targets 是数量，不能原样作为 PUT 的 targets。重命名 deployment 后须重建其副本，确保实际 served model 与容器名称一致。

Qwen 保留固定镜像、262144 上下文、TP4 + EP、GPU PLE、FlashInfer 自动调优、MTP 3 和 NCCL。Nemotron 保留 BF16、pooling、32768 上下文、4096 维和 query/document 语义；Qwen3-Embedding 不能混入 Nemotron 副本池。MinerU 使用自己的固定镜像与参数，不套用 Qwen 镜像。共卡服务分别配置显存预算和通信端口，并测量联合峰值。

## 验收与维护

```bash
bash deploy/gpustack/manage.sh check --model nemotron-embed
```

check 只认证读取模型列表，不证明推理、分流或恢复。发布还需验证：

- Qwen 文本、角色、SSE、工具、图片与 thinking；Embedding 新旧向量、满长不截断输入和检索排序。
- 真实 PDF 从业务 API 走 MinerU 主路由，核对所有节点及每个 DP rank 的请求增量；同时检查两台应用的实际生效配置。
- 维护下线单副本后主入口仍可用，恢复后重新参与分流。该验证不等同于突然断电或控制面故障切换。

备份控制面先停服务，再冷备整个持久卷（包含数据库与网关状态），备份权限 600。变更项目或卷名时将旧卷完整复制到新卷并比较，沿用私有凭据与绑定地址；启动后确认原 worker、模型、route ID 和 API key 均保留，再删除旧容器及旧卷。不要运行全局 prune。

需要恢复原生部署时先将对应 GPUStack deployment 缩为 0，确认模型子容器与 GPU 释放，再恢复原生实例；只停止 worker 或控制面并不能保证子容器停止。已删除的旧编译缓存可重新生成，运行中的模型权重和卷必须保留。

## 上游依据

- [固定版本发行说明](https://github.com/gpustack/gpustack/releases/tag/v2.2.3)
- [Docker 安装](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/installation/installation.md)
- [模型路由](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/user-guide/model-route-management.md)
