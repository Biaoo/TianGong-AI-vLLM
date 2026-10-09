# 架构

## 模型与机器分离

```text
共享模型配置 deploy/profiles/*.json ─┐
                                    ├─ manage_models.py ─ Docker Compose ─ 实例 A
私有机器实例 deployments/*.json ────┘                                  └─ 实例 B
模型清单 ─ download_model.py ─ 校验后的权重目录（只读挂载，可复用）
```

模型配置定义镜像、权重身份、默认启动参数和验收脚本；机器实例定义 GPU 列表、并行规模、端口、上下文、显存预算和目录。同一模型可部署多个实例，新增同类模型通过配置注册；专用 SDK 可使用 replicated-http 运行方式。具体字段见[多模型管理](models.md)。

推理引擎负责 tokenizer、原生模型模板、内核、批处理和 OpenAI-compatible API。Docker Compose 负责实例容器、网络、缓存卷、日志轮转和退出恢复。无需额外代理、Redis、数据库或 PM2。

## 生命周期与隔离

统一入口 `deploy/manage.sh` 从任意工作目录定位仓库。默认项目名 `tiangong-<实例名>`，容器服务名 inference、内部 API 端口 8000；不同实例独立选择主机端口和 GPU。模型目录只读，缓存卷按项目隔离。GPU 可以显式共享，但必须人工核对总显存预算，此工具不自动调度 GPU。

start 不重建已有容器；restart 应用新配置；stop 保留模型、缓存和证据。all 仅枚举本机 deployments，不自动启动模型目录中的全部模型配置。旧 .env 生成模型保留 tiangong-vllm/model 的兼容身份，通过显式 model 操作。

## 文件与接口

- `deploy/profiles/`：镜像、默认参数、下载清单路径和模型验收脚本。
- `deploy/examples/`：不含主机私有路径的实例模板；`deployments/`：私有部署实例。
- `deploy/vllm/`：Qwen3.8 构建修复和模型清单；`deploy/embedding/`：Qwen3 embedding 清单。
- `.env`：私有密钥及旧生成模型机器配置；`models/`：私有权重目录。
- `scripts/`：标准库实现的配置管理、下载、完整性校验与真实 API 验收。
- `output/validation/`：真实响应、耗时、失败原因等私有证据。

Embedding 客户端调用 `/v1/embeddings`；生成模型客户端调用 `/v1/chat/completions`。不因引擎暴露某条路由就视为通过验收，每个模型配置指定独立 check_script。旧模型模板和生成专用参数不自动套用到新模型。

## 每卡独立 HTTP 副本

Laya 采用一个机器实例管理多个 worker 和一个 Nginx 网关。gpus 列表中的每张卡分配一个完整模型容器，仅开放容器内部端口；统一网关在宿主发布 7720，以最少连接数调度。每个 worker 一个 Python 模型进程和单独请求队列，不进行跨 GPU 模型切分。该运行方式由通用管理器的 replicated-http 实现，模型专用逻辑在容器入口与 check_script 中。
