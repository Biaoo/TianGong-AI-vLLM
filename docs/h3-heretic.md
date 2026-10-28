# H3 Heretic 视频接口

该版本用 Heretic Qwen3-VL 编码参考图和提示词，后续生成继续使用 FastH3 Omni NVFP4 QAD、8 步采样和 36 层视频解码器。它是独立的 Ref2VA deployment，与原 H3 和 FL2VA 路由并存。

## 模型组成

- 编码器来自 [ethanfel 的 H3 模型仓库](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)，固定 revision 与 SHA256 见 `deploy/h3-heretic/release.json`。
- 使用其中的 BF16 源文件，保留语言层 0–49 和完整视觉塔。打包脚本只转换张量名称，BF16 数据逐字节保留；运行时对编码器执行在线 FP8。没有加载 ComfyUI 的 INT8 ConvRot 权重。
- Heretic 改动位于语言层 31–40 的 attention output projection。它改变 H3 的条件编码；作者的聊天拒答测试不是视频质量测试。
- 不需要用于独立文本生成的 50–63 层 tail、final norm 或 LM head。
- 扩散模型、参考图处理器、tokenizer 和 VAE 继续使用 [H3 固定版本](h3.md)，无运行依赖升级。

## 准备与部署

1. 按 release.json 下载 BF16 源文件并校验。准备已有固定 FastH3 快照与原生 H3 FL2VA 快照。
2. 使用 `scripts/prepare_h3_heretic.py --source <BF16文件> --fast <FastH3快照> --native <原生快照/FL2VA> --output <新目录>`。脚本核对全部 902 个张量的名称、精度与形状，复用既有视频组件，生成 `assembly-manifest.json`。
3. 将输出的 manifest SHA256 写入 release.json，执行 `bash deploy/h3-heretic/build.sh`。启动器先校验固定 manifest、每个组件和原生 VAE，再离线加载。
4. 在 GPUStack 创建 `deploy/examples/h3-heretic-gpustack-backend.json` 和 model 模板对应的私有实例，设置单卡选择、只读模型根目录、32 GiB claim、0.30 allocator 上限和允许的网关 IP。模板初始副本为 0；权重、镜像就绪后启动 1 个副本。
5. 为它创建独立 generic-proxy 路由；不把该模型加入原 FastH3 的负载均衡池。使用 `scripts/check_h3.py` 通过实际路由验收，并保存原版同图、同提示词、同 seed 的对比视频。

权重组装目录与引用的 FastH3 文件须位于同一模型挂载根下。GPUStack 管理生命周期，本版本不加入原生 `all`。

## 调用

接口为 `POST <GPUStack 路由地址>/v1/videos/sync`，字段与原 FastH3 相同：`prompt`、`input_reference`、`seconds`、`size`、`seed`。内网或公网鉴权策略在路由层配置；公网专用 key 仅授权对应 H3 路由。

```bash
curl --fail-with-body "$H3_HERETIC_URL/v1/videos/sync" \
  -H "Authorization: Bearer $H3_API_KEY" \
  -F 'prompt=The cat and mouse wave at each other, then walk together across the room. Soft footsteps and cheerful music.' \
  -F 'input_reference=@reference.png' \
  -F 'seconds=5' -F 'size=832x480' -F 'seed=2102' \
  --output heretic.mp4
```

首尾帧约束继续使用独立 FL2VA 接口。移除本版本时只停用对应路由和 deployment，保留既有 H3 服务及共享权重。
