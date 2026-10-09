# 多模型与机器实例管理

## 两层配置

模型配置 `deploy/profiles/<名称>.json` 描述模型本身：`model_id`、固定镜像 `image`、逐文件 SHA256 清单 `manifest`、`entrypoint`、默认 vLLM `options`、容器 `environment`、可选 `build` 和 `check_script`。可以是 embedding、生成、视觉、重排等 vLLM 支持的模型；新增同类模型不需要修改管理脚本。runtime 默认为 vllm；专用 HTTP SDK 使用 replicated-http，为每张 GPU 创建独立完整副本并通过固定 Nginx 镜像分发请求，参考 Laya 配置。未配置 `check_script` 的模型可下载和启动，但 `check` 会明确失败，不能将健康检查当作验收。

本机实例 `deployments/<实例名>.json` 引用模型配置，保存机器差异。实例名决定默认 Compose 项目名 `tiangong-<实例名>`。同一个模型配置可供多台机器使用，也可以在同一台机器上运行多个实例。实例和 `.env` 保持私有；可复用、不含主机路径的模板放在 `deploy/examples/`。

```json
{
  "profile": "qwen3-embedding-8b",
  "model_dir": "models/Qwen3-Embedding-8B",
  "gpus": ["0", "1", "2"],
  "gpu_mode": "nvidia",
  "host": "0.0.0.0",
  "port": 7710,
  "api_key_env": "EMBEDDING_API_KEY",
  "options": {
    "data-parallel-size": 3,
    "tensor-parallel-size": 1,
    "max-model-len": 40960,
    "gpu-memory-utilization": 0.60
  }
}
```

单卡机器改为 `gpus: ["0"]`、`data-parallel-size: 1`；根据显存选择上下文长度和显存比例。张量并行则调整 `tensor-parallel-size`；支持的模型也可配置 `pipeline-parallel-size`。管理器检查 TP × DP × PP 不超过分配的 GPU 数量。这只是拓扑检查，模型是否支持该并行策略及显存是否足够仍须真实启动验证。

## 实例字段与参数覆盖

| 字段 | 用途 |
| --- | --- |
| `profile` | `deploy/profiles/` 下的配置名称，不带扩展名 |
| `model_dir` | 本机模型目录；相对路径从仓库根目录解析；可复用校验过的缓存 |
| `gpus` | 任意长度的 GPU ID/UUID 列表，不会隐式分配全部 GPU |
| `gpu_mode` | `cdi` 使用 Docker 原生 CDI；`nvidia` 使用 NVIDIA runtime、设备预留和 UVM，适用于已配置此 runtime 的 Snap Docker |
| `host` / `port` | 对外监听地址和端口；容器内统一 8000 |
| `options` | 覆盖模型配置的 vLLM 命令行参数，不带 `--` |
| `image` | 可选的实例级镜像覆盖；升级后须重新验收，建议固定摘要 |
| `served_model_name` | 可选的客户端模型名覆盖 |
| `environment` / `shm_size` | 实例级容器环境和共享内存配置 |
| `api_key_env` | 从本机 `.env` 或进程环境读取的密钥变量名；为空或未设置时不启用鉴权 |
| `project_name` / `service_name` | 可选的既有部署身份；新实例通常无需设置，不能与其他实例重复 |

`options` 是浅层覆盖：标量替换默认值，JSON 对象作为单个参数序列化，数组展开为多个参数，`true` 添加开关，`false` 输出 vLLM 的 `--no-<名称>`，`null` 移除默认参数。只对支持否定形式的 vLLM 开关使用 false。模型位置、监听地址、端口、模型名和 API key 由专门字段管理，不能混入 options。

例如生成模型用 `"speculative-config": null` 禁用配置中的 MTP；embedding 不添加生成模型专用的 MTP、工具解析或量化参数。上下文上限不能超过模型和引擎支持范围；pooling 的批处理 token 预算也要覆盖实际最长输入。

## 管理命令

```bash
./deploy/manage.sh profiles
./deploy/manage.sh init embedding-single embeddings-small
# 编辑 deployments/embeddings-small.json
./deploy/manage.sh config embeddings-small
./deploy/manage.sh pull embeddings-small
./deploy/manage.sh download embeddings-small
./deploy/manage.sh verify embeddings-small
./deploy/manage.sh start embeddings-small
./deploy/manage.sh status embeddings-small
./deploy/manage.sh check embeddings-small
./deploy/manage.sh logs embeddings-small
./deploy/manage.sh stop embeddings-small
```

`init <示例名> <实例名>` 不覆盖已有实例。`list` 只列本机实例元数据，不输出密钥。`config` 只验证配置，不打印展开后的环境。Compose 配置在内存中生成并经标准输入传给 Docker；密钥不出现在命令行或公开配置文件。

`start` 不重建容器，配置变更使用 `restart`；`start-loaded` / `restart-loaded` 禁止构建和拉取。`build` 对带 build 的配置构建镜像，否则拉取固定镜像。`all` 按顺序操作本机实例文件，不自动启动所有模型配置，也不包含旧 `.env` 的隐式 model。流式日志必须指定一个实例。

每个实例有独立容器、网络和缓存卷；权重只读挂载，可让多个实例复用同一份权重。停止不删除模型或缓存。启动前人工确认实例端口不冲突、所有共享 GPU 服务的显存预算合理；此工具不是跨机器调度器，不会自动迁移负载或处理端口冲突。

## 新模型与升级

1. 在 `deploy/profiles/` 添加配置，固定适配该模型的 vLLM 镜像；需要定制时提供独立 Dockerfile。
2. 为 ModelScope 模型固定 revision 和文件列表，清单格式参考已有 `model-manifest.json`。下载器在对应固定推理镜像的临时容器内运行（不分配 GPU，以宿主 UID/GID 写入文件），按清单校验大小和 SHA256，支持断点下载；verify 将目录只读挂载，不写锁文件。先 pull/build 镜像，再 download/verify；当前内置下载源为 ModelScope，其他来源须先准备权重和相应校验流程。
3. 新建不含机器私有信息的实例模板，再通过 init 创建本机实例，配置 GPU、路径与端口。
4. 为模型提供真实验收脚本，检查其接口、输出、鉴权和错误行为。升级在新模型目录或固定新镜像中进行，保留可回滚的旧资源。

迁移至其他机器时，同步 Git 中的模型配置和清单，传输/下载权重与固定镜像；在目标机器创建自己的实例配置，再执行 verify、start-loaded 和 check。机器配置独立于模型版本，不能直接照搬源机器的 GPU 数量或显存参数。

## 旧 Qwen3.8 部署兼容

当 `deployments/model.json` 不存在且 `.env` 存在时，`manage.sh <action> model` 将原 `.env` 参数转换为实例配置，保留项目 `tiangong-vllm`、服务 `model` 和缓存卷 `vllm-cache`。显式实例优先于 `.env` 的机器参数；该隐式实例不属于 all，停止它需 `stop model`。

把既有部署改成实例文件时，设置 `project_name: "tiangong-vllm"` 和 `service_name: "model"`，并迁移全部参数后再使用 restart。旧 `deploy/vllm/compose.yaml` 仅保留为兼容参考；新管理入口由模型配置生成 Compose。

## 专用 SDK 的多副本运行

`runtime: "replicated-http"` 使用配置的 entrypoint 与 options 启动 worker，保持 /model、--host、--port、--served-model-name 的统一参数约定。每张 GPU 分配一个容器，GPU 列表长度决定副本数；不接受 TP/DP/PP 参数。模型配置还需 gateway_image 指定固定 Nginx 镜像。新 SDK 服务须提供 /health 并实现实例 api_key_env 对应的 VLLM_API_KEY 环境变量鉴权。

管理器将下载、构建、生命周期和 check 统一到同一实例名；check_script 可读取 MODEL_BASE_URL、MODEL_ID、MODEL_API_KEY、MODEL_REPLICA_COUNT。只有网关发布主机端口，副本各有独立缓存卷。扩缩容修改 gpus 后执行 restart；运行细节见 [Laya](laya.md)。
