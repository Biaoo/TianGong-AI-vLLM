# Qwen3 Embedding Docker 部署

## 模型与实例

模型配置为 `deploy/profiles/qwen3-embedding-8b.json`。默认使用官方 vLLM 0.29.0 固定摘要镜像，显式 pooling runner，模型名 `Qwen/Qwen3-Embedding-8B`，4096 维单位向量。该 ModelScope 快照缺少 Matryoshka 声明，因此模型配置显式设置 `hf-overrides: {"is_matryoshka": true}`，启用模型本身支持的 dimensions；不修改只读权重。模型清单固定在 `deploy/embedding/model-manifest.json`，使用 ModelScope revision `fcd9221ff0c460fdd81bfd04f4bc5b014e58da5b`。

`deploy/examples/embedding.json` 是三卡 DP3/TP1、7710 端口、40960 上下文、显存比例 0.60 的机器模板；`embedding-single.json` 展示单卡和较短上下文。同一模型在其他机器上应创建自己的实例文件，详见[多模型管理](models.md)。模板不是所有机器的推荐显存配置，须核对与其他服务的共享预算。

```bash
./deploy/manage.sh init embedding embedding
# 编辑 deployments/embedding.json
./deploy/manage.sh config embedding
./deploy/manage.sh pull embedding
./deploy/manage.sh download embedding
./deploy/manage.sh start embedding
./deploy/manage.sh status embedding
./deploy/manage.sh check embedding
```

已有权重时把 model_dir 指向缓存目录，并执行 verify；不会重下载已验证文件。镜像包含 CUDA、Torch、Transformers 和 vLLM，不再向宿主安装推理库。NVIDIA runtime 模式显式指定 runtime、GPU 设备预留和 UVM，CDI 主机可使用 `gpu_mode: "cdi"`。

## 从原生 PM2 迁移

1. 确认 PM2 进程的工作目录、命令、GPU、端口和调用方；保存该进程配置及旧 `.venv` 供回退。仅操作本项目的 embedding 进程。
2. 下载固定镜像、运行容器 CUDA 检查、验证全部模型文件；在旧 API 上保存固定中英文样本向量，私有基线格式见[验证说明](validation.md)。
3. 配置实例并检查 Compose；确认指标中 running/waiting 为零、客户端连接已排空后，停止对应 PM2 进程，启动 Docker 实例。相同端口切换存在加载和编译中断窗口。
4. 执行 `manage.sh check embedding`，另用 `python3 scripts/check_embedding.py --baseline <基线文件>` 对比相同输入的新旧向量。该脚本直接执行时从 `.env`/进程环境读取 EMBEDDING_HOST、EMBEDDING_PORT、EMBEDDING_SERVED_MODEL_NAME、EMBEDDING_API_KEY；通过管理入口执行则自动使用实例配置。
5. 验收通过后，仅移除旧 embedding PM2 项并保存 PM2 列表，避免机器重启时双重拉起。Docker 使用 `unless-stopped`；保留旧配置、旧环境和验收证据以便回退。

回退时先停止 Docker 实例，确认端口和对应 GPU 内存已释放，再使用备份的原生配置恢复同一个 PM2 服务；不能两者同时占用相同端口/GPU 预算。开发依赖使用 `.venv-dev`，迁移期间不清理原 `.venv`。

## 调用

```http
POST http://127.0.0.1:7710/v1/embeddings
Content-Type: application/json

{"model":"Qwen/Qwen3-Embedding-8B","input":["生命周期评价用于量化产品环境影响。","Life cycle assessment measures environmental impacts."],"encoding_format":"float"}
```

默认无需鉴权；通过实例 `api_key_env` 引用非空密钥后，客户端发送 Bearer Authorization。使用 `dimensions: 1024` 可返回较低维向量；也支持 base64 输出。检索 query 根据[模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)添加任务指令，文档段落不强加 query 指令。

保持已有向量库的模型、维度和输入预处理一致。迁移验收覆盖有限中英文样本的新旧余弦相似度，不代替业务语料检索质量评估或满长并发测试。
