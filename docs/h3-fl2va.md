# H3 首尾帧音视频服务

使用原版 MiniMax H3 的 `FL2VA` 分区，接受首帧、尾帧或同时提供两帧，生成连接画面的同步音视频。它与 [FastH3 NVFP4 Ref2VA](h3.md) 使用独立的 GPUStack backend、deployment 和路由。

## 固定运行配置

[release.json](../deploy/h3-fl2va/release.json) 记录源码、模型 revision、镜像摘要和验收状态。配置为单卡、在线 FP8、DiT 与文本编码器逐层卸载、cuDNN attention、50 步、guidance 1、并发 1。视频/音频 VAE 保留原精度。下载文件仍为原版权重，共约 134.16 GiB；磁盘大小不代表推理显存。

FP8 与首尾帧参数按固定 [vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f/recipes/MiniMaxAI/MiniMax-H3.md) 配置。运行时继承已固定的 H3 基础镜像，不安装或升级依赖。构建脚本核对基础镜像 ID，并拒绝覆盖已有标签：

```bash
bash deploy/h3-fl2va/build.sh
```

## 下载与启动

创建 `/srv/models/h3/fl2va/9bfb6693f2cf6de171db46d1aa586f67d773a1da/`，将 [模型清单](../deploy/h3-fl2va/model-manifest.json) 复制到其中的 `snapshot-manifest.json`，然后下载：

```bash
python3 scripts/download_hf_snapshot.py \
  --model MiniMaxAI/MiniMax-H3 \
  --revision 9bfb6693f2cf6de171db46d1aa586f67d773a1da \
  --directory /srv/models/h3/fl2va/9bfb6693f2cf6de171db46d1aa586f67d773a1da \
  --include-prefix FL2VA/ --include-prefix model_index.json \
  --transport modelscope --mirror-model MiniMax/MiniMax-H3 \
  --mirror-revision 57559a6720ddaa2eaee52b09ec2ef011127997b2 \
  --ipv4 --workers 16
```

下载器按固定 Hugging Face 清单验证 ModelScope 文件，保留断点；启动器再次离线验证整个快照。已存在的 VAE 文件只有在大小与 SHA256 都一致时才可复用。

注册 [backend](../deploy/examples/h3-fl2va-gpustack-backend.json) 和 [model](../deploy/examples/h3-fl2va-gpustack-model.json) 模板；替换实际网关、worker、GPU 和模型路径。模型路径指向快照中的 `FL2VA`，专用模型父目录须已挂载到 worker 和模型容器。启动器将该 bind mount 设为只读；不得对包含其他可写数据的共享挂载执行此操作。

模板初始副本为 0，GPUStack 显存 claim 为 32 GiB，PyTorch allocator fraction 为 0.30，SHM 为 16 GiB。下载和校验完成、核对同卡显存及主机内存后启动一个副本。allocator 不覆盖所有 CUDA 分配，真实峰值以生成采样为准。独立 generic-proxy 路由按部署的内网访问策略设置；不加入原生管理器的 `all`。

## 首尾帧 API

`POST <GPUStack 路由前缀>/v1/videos/sync`，返回 MP4。`input_references` 重复上传两次，顺序为首帧、尾帧；`frame_indices` 与图片一一对应。

```bash
curl --fail --show-error --noproxy '*' --max-time 7200 \
  'http://<controller>:17880/model/proxy/<route-id>/v1/videos/sync' \
  -F 'input_references=@./first.png;type=image/png' \
  -F 'input_references=@./last.png;type=image/png' \
  --form-string 'prompt=主体从首帧的位置自然移动到尾帧的位置，动作连贯，保持外观一致，配合环境音，无字幕。' \
  -F 'extra_params={"task":"fl2va","frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F 'flow_shift=12' -F 'num_inference_steps=50' \
  -F 'size=832x480' -F 'seconds=5' -F 'seed=2102' \
  --output video.mp4
```

只给首帧时上传一张图并用 `[0]`；只给尾帧时上传一张图并用 `[-1]`。`seconds` 为 4～15 的整数，`size` 可选 `832x480` 或 `1344x768`。

## 生成验收与恢复

在带 ffmpeg/ffprobe 的固定 H3 镜像中运行：

```bash
python3 scripts/check_h3_fl2va.py \
  --base-url http://<controller>:17880/model/proxy/<route-id> \
  --first /path/to/first.png --last /path/to/last.png \
  --output-dir output/h3-fl2va-acceptance/run-unique
```

脚本验证尺寸、时长、24 FPS、32 kHz 双声道和完整媒体解码，保存请求参数、输入散列、耗时及生成视频首尾帧。人工对比输出首尾帧与输入，再记录视觉检查结论；媒体解码通过不等于首尾约束验收。需要鉴权的路由使用 `--key-stdin`，凭据不进命令或结果文件。

生成时采样显存与主机内存，并核对同卡原有模型请求、容器 ID 和启动时间。失败时只将新 FL2VA deployment 设为 0，保留模型和证据；原有 FastH3 路由与副本保持原配置。恢复前核对固定镜像和快照，单副本生成通过后再对外提供路由。
