# 依赖管理

## 隔离边界

模型推理使用 `deploy/profiles/` 固定的镜像，机器实例通过通用管理器生成 Docker Compose；CUDA、Torch、vLLM、Transformers 及推理内核全部留在镜像内。宿主仅依赖 NVIDIA 驱动、Container Toolkit base 1.20.0、支持 CDI 的 Docker 28.3+ / Compose 2.24.4+与 Python 3.12+。下载/校验在对应镜像的临时容器内使用 Python 标准库和 curl，无需给宿主安装推理库。

`pyproject.toml` 无应用运行依赖；`uv.lock` 固定 Black、Ruff、pytest 等开发工具。标准库脚本可以直接 `python3` 执行。不要向宿主环境重新添加 Torch 或 vLLM，不修改系统 Python。

## 更新方法

1. 模型文件由 `model-manifest.json` 固定 ModelScope revision、大小和 SHA256；更新时在新目录下载校验，不能原地覆盖运行中的权重。
2. 基础镜像使用验证过的 digest，在其上构建受 SHA256 检查约束的运行时修复层，见[运行时修复](runtime-patches.md)。切换镜像必须核对架构、混合量化、CUDA SM120 兼容性，再完成真实 HTTP 验收。
3. 开发工具通过 `uv lock --upgrade` 更新，并提交锁文件；`uv sync --locked --group dev` 安装。锁文件使用官方 PyPI 解析。迁移宿主原生服务时设置 `UV_PROJECT_ENVIRONMENT=.venv-dev`，避免清理仍用于回滚的旧 `.venv`。
4. 记录验证条件和已知边界，成功后更新示例配置及相关指南。保留需要回退的镜像和模型；不要全局 prune。

来源：[模型卡](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)、[vLLM 仓库](https://github.com/vllm-project/vllm)。

## 已固定的引擎基线

基础镜像 `vllm/vllm-openai@sha256:dea7fa047caa114167efccdeb42321b12d2c11da1add8728aa2662cb8d6b8cd5`，vLLM `0.29.1rc1.dev347+gdee37d891`，构建提交 `dee37d89115db4c94a820a79a78a7828e141c910`，Torch `2.13.0+cu130`、CUDA 13.0、Transformers 5.17.0。它是开发构建，因此固定摘要而不在启动时追踪 nightly。

## Embedding 引擎

Embedding 独立使用官方 vLLM 0.29.0 稳定版镜像，固定多架构摘要 `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，amd64 摘要 `sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`。推理依赖整体随镜像升级，不在镜像启动时 pip 安装或升级 Torch、Transformers、CUDA。镜像内实测 Torch 2.13.0+cu130、Transformers 5.16.1、FlashInfer 0.6.18、CUDA 13.0；生成模型仍使用上面的独立基线。

开发锁文件更新至 Black 26.5.1、Ruff 0.16.8、pytest 9.1.1 及解析时可用的兼容间接依赖。GPU 设备注入方式、模型清单和验收步骤见 [Embedding 部署](embedding.md)。官方参考：[vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)、[Docker 部署](https://docs.vllm.ai/en/stable/deployment/docker/)、[Pooling 模型](https://docs.vllm.ai/en/stable/models/pooling_models/)。

## Laya 专用 SDK 镜像

`deploy/laya/Dockerfile` 复用 embedding 的固定 vLLM 0.29.0 基础镜像作为 Blackwell-compatible Torch/CUDA 运行环境，额外安装 SHA256 固定的 laya 0.3.4 wheel，使用 --no-deps 保持基础依赖版本。启动入口为专用 HTTP 服务，不运行 vLLM 推理引擎。Nginx 1.31.1 网关独立固定摘要，模型配置声明 runtime=replicated-http。完整拓扑、接口和副本参数见 [Laya 部署](laya.md)。
