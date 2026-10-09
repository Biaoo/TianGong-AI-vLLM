# GPUStack 可选试用

本项目维护模型镜像、权重、参数和验收；GPUStack 提供独立的管理界面、调度器与统一推理网关。试用只覆盖模型部署管理和请求负载均衡，文档解析 API、Celery、任务队列及结果存储仍由其业务项目维护。

## 版本与隔离

控制面镜像固定为 GPUStack v2.2.3，发行索引摘要保存在 `deploy/gpustack/release.json`。升级时重新核对稳定版本、摘要与接口，不使用浮动 latest，也不将 GPUStack 安装进宿主 Python 环境。此组件使用 GPUStack 自带的数据库与网关，不改变原生模型管理器的运行依赖。

`deploy/gpustack/manage.sh` 是独立控制面入口。私有配置位于 `deployments/gpustack/trial.json`，不会被原生 `deploy/manage.sh ... all` 枚举；Compose 项目固定为 `tiangong-gpustack-trial`，独立持久卷。控制面不挂载 Docker socket、不分配 GPU，并显式禁用内嵌 worker。启动控制面不会登记旧模型、启动模型副本或切换业务流量。

推荐将控制面放在日常管理主机，将 worker 放在实际提供 GPU 的服务器。每台 GPU 主机通常注册一个 worker，由 GPUStack 对该 worker 上的模型实例选择 GPU；不需要每张卡安装一个管理 worker。纯 Provider 代理阶段无需 worker。控制面的数据库与配置需持久保存；管理主机维护会影响统一入口的可用性，不能把多个模型副本等同于控制面高可用。

## 启动控制面

在准备运行控制面的主机上，从仓库根目录执行：

```bash
bash deploy/gpustack/manage.sh init
# 按需编辑 deployments/gpustack/trial.json 的 host 和 port
bash deploy/gpustack/manage.sh config
bash deploy/gpustack/manage.sh pull
bash deploy/gpustack/manage.sh start
bash deploy/gpustack/manage.sh status
```

需要 Docker Compose v2。Ubuntu 尚未安装 Docker 时，可先执行 `bash deploy/gpustack/install-docker-ubuntu.sh --check`，再在原主机终端执行 `sudo bash deploy/gpustack/install-docker-ubuntu.sh --install`。安装器使用 Ubuntu 的 docker.io/docker-compose-v2 包，预先设置 `ip-forward-no-drop=true`，保留其他 daemon 配置，避免 Docker 将转发策略改为 DROP；已有 Docker 或冲突配置时拒绝覆盖。安装涉及系统服务与 Docker 组权限，新登录后生效；安装器不启动 GPUStack。

默认绑定本机 `127.0.0.1:17880`，可通过 SSH 转发访问；需要其他主机的 worker 接入时改为控制机的明确 LAN IPv4 地址，并确认 server 与 worker 双向可达。控制面可运行在无 GPU 的机器上；放在哪个代码仓库与运行在哪台主机是两个决定。

`init` 只创建私有配置及 mode 600 的 `.secrets/gpustack-trial/bootstrap-password`，不覆盖已有文件。登录用户名为 admin，首次密码由该文件提供；在 UI 修改密码后，文件只是初始化资料。`config` 仅调用 Compose 的静默校验，不打印展开配置或凭据。

`pull` 是显式下载；`start` 不拉取、不构建、不重建已有容器。修改端口或其他配置后执行 `restart`，只重建试用控制面，数据库卷保留。`stop` 保留容器与数据，不提供自动删除卷或清理其他服务的命令。命令不会打印 Docker 错误原文；故障时在原主机查看容器状态与日志，避免将凭据带入共享记录。

## 第一阶段：现有接口接入网关

在 GPUStack UI 的 Providers 中为每个现有上游建立 Generic Provider，配置实际 API 地址、模型标识和需要的鉴权。在 Routes 中建立独立的试用模型别名，将多个等价上游加入同一 route，设置权重及需要的 fallback。确认 Generic Provider 的 base URL 与路径拼接符合部署版本；只有真实请求验证通过后才作为可用路由。

- 外部 Provider 接入只提供 API 代理与流量分配，不接管旧容器的部署、重启或 GPU 分配。
- Embedding 副本必须使用相同权重、维度、归一化和输入预处理。不能把 Nemotron 与仓库中的 Qwen Embedding 配置混成一个可互换的副本池。
- `/v1/embeddings` 与 `/v2/embed` 的协议不同，尤其是 query/document 的提示词语义。使用后者的客户端要单独验证 Generic Proxy 或保留兼容层，不直接改成前者。
- MinerU 只接入 VLM 推理接口，保留实际镜像和专用参数。仅监听上游主机回环地址的服务不能被远端控制面直接访问；需先规划可控的内网可达方式。
- 权重控制请求份额，不代表保证两台机器 GPU 使用率相同；先测吞吐和延迟，再决定权重。

试用客户端显式使用新入口及试用模型名，原调用方继续使用原入口。不要为了测试 fallback 停止生产模型；使用独立试验副本或仅调整试用路由目标。

## 第二阶段：GPUStack 原生托管

已验证Docker原生CDI的主机可使用[worker试点入口](gpustack-workers.md)；其他运行时须单独适配。

在 UI 的 Clusters / Workers 中生成与当前版本匹配的 worker 注册配置，确认 Docker/NVIDIA runtime、实际 GPU 列表及可用资源后部署。worker 能创建模型容器，所需 Docker 权限与控制面不同，不能把控制面容器直接改成全 GPU worker。

以一个独立模型实例开始：

1. 保留实际模型 ID、固定权重和已验证镜像；需要项目补丁时使用自定义推理后端，不自动换成 GPUStack 默认 vLLM 镜像。
2. 创建试用 deployment，显式选择 worker/GPU、每副本 GPU 数、上下文、批量与显存参数。四卡张量并行是一个副本，不能当四个可独立均衡的副本；DP 组与平台副本层次也需要区分，避免重复扩容。
3. 同卡 Embedding 与 MinerU 分别设预算和并发，并验收联合峰值。显存比例不是硬件隔离；实例切换前排空原服务，不能让原 Compose/PM2 与 GPUStack 同时启动同一服务。
4. 模型实例就绪后建立试用 route，验证模型输出、重启恢复、请求分布与故障切换。
5. 通过后再逐个迁移调用方。原生管理器保留为独立部署方式；同一实例只有一个生命周期管理者。

已有模型配置可作为人工填写 GPUStack 的参数依据，但本功能不把本机实例 JSON 冒充 GPUStack 导入格式。Worker 身份、设备 ID 和后端配置以已安装版本为准。

## 检查与验收

在 UI 为试用 route 创建 API key，将它保存为 mode 600 的 `.secrets/gpustack-trial/api-key`，然后执行：

```bash
bash deploy/gpustack/manage.sh check --model embedding-trial
```

该命令仅认证读取 `/v1/models`，检查指定 route 对当前 key 可见；不提交推理，不证明后端健康、向量一致性、多个副本被使用或故障切换成功。HTTP 重定向被拒绝，避免把 key 转发到其他地址。

上线前还需要分别完成：

- 对新旧入口发送相同合成输入，对比 Embedding 维度、归一化、相似度和检索排序；验证 32K 等实际长度要求及 `/v2/embed` 语义。
- 生成模型验证文本、SSE、工具、图片、thinking 和实际上下文；MinerU VLM 验证专用请求参数和有效输出。
- 用后端请求计数或路由指标证明至少两个目标实际接收请求；仅得到若干次 HTTP 200 不能证明均衡。
- 独立试验副本不可用时验证 fallback、超时和恢复；流式响应已经输出后不能无缝重放。分别记录模型与控制面可用性，单个控制面仍是统一入口的单点。

现有直连模型验收脚本可能读取 `/health`、`/version` 等后端专用接口，不能假定改 URL 就能通过网关运行；模型质量验收与网关合同验收分开执行。本功能的单元测试和 Compose 校验不替代上述实机验证。

## 退出试用

只接入 Provider 时，让试用客户端恢复原入口后执行 `bash deploy/gpustack/manage.sh stop`。已创建托管实例时，先在 GPUStack 停止对应 deployment 并核对 worker 上资源释放，再恢复原实例；停止控制面不等于停止 worker 上的模型。保留试用数据库卷供恢复，不使用 Docker prune。

## 上游依据

- [v2.2.3 发行版](https://github.com/gpustack/gpustack/releases/tag/v2.2.3)
- [固定版本 Docker 安装](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/installation/installation.md)
- [固定版本启动参数](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/cli-reference/start.md)
- [Provider](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/user-guide/model-provider-management.md) / [模型路由](https://github.com/gpustack/gpustack/blob/v2.2.3/docs/user-guide/model-route-management.md)
