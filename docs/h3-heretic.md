# H3 Heretic 视频接口

Heretic Qwen3-VL 用于编码图像和提示词，有两种独立 deployment：FastH3 Omni NVFP4 QAD 的 8 步 Ref2VA 参考图生成，以及原生 H3 FP8 的 50 步 FL2VA 首尾帧／T2VA 纯文本生成。两种服务与各自原版并存，分别使用独立 GPUStack 路由。

## 模型组成

- 编码器来自 [ethanfel 的 H3 模型仓库](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)，固定 revision 与 SHA256 见 `deploy/h3-heretic/release.json`。
- 使用其中的 BF16 源文件，保留语言层 0–49 和完整视觉塔。打包脚本只转换张量名称，BF16 数据逐字节保留；运行时对编码器执行在线 FP8。没有加载 ComfyUI 的 INT8 ConvRot 权重。
- Heretic 改动位于语言层 31–40 的 attention output projection。它改变 H3 的条件编码；作者的聊天拒答测试不是视频质量测试。
- 不需要用于独立文本生成的 50–63 层 tail、final norm 或 LM head。
- 扩散模型、参考图处理器、tokenizer 和 VAE 继续使用 [H3 固定版本](h3.md)，无运行依赖升级。
- 新镜像对固定上游源码增加 CPU 初始化修正：编码器在 GPU 逐层量化时，已量化的 NVFP4 DiT 继续在 CPU 加载，避免两者在启动时同时占用显存。补丁先检查上游文件 SHA256，只作用于已配置卸载的 NVFP4 DiT。

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

移除本版本时只停用对应路由和 deployment，保留既有 H3 服务及共享权重。

## Heretic FL2VA 与 T2VA

`deploy/h3-heretic-fl2va/` 使用固定原版 FL2VA 镜像作为父镜像，继承在线 FP8、逐层卸载、50 步和原生 VAE；替换编码器为同一套已转换 Heretic BF16 源权重。没有下载第二份扩散模型或修改现有 deployment。

准备原生组合：

```bash
python3 scripts/prepare_h3_heretic.py \
  --native-assembly-from /srv/models/h3/heretic-e8967f6a-fastnvfp4-v1 \
  --expected-manifest-sha256 8d01e4edca0993969cd2991d7eebbc3a9101438f6003162cc265f9b78d18f1bf \
  --native /srv/models/h3/fl2va/9bfb6693f2cf6de171db46d1aa586f67d773a1da/FL2VA \
  --output /srv/models/h3/heretic-fl2va-e8967f6a-v1/FL2VA \
  --allowed-root /srv/models/h3
bash deploy/h3-heretic-fl2va/build.sh
```

组装脚本核对来源散列、原生分区任务声明和各组件，替换整个 `text_encoder/`，其余组件使用同挂载内链接。输出 manifest digest 必须与 `deploy/h3-heretic-fl2va/release.json` 一致；启动时重新校验后离线加载。

GPUStack 模板为 `deploy/examples/h3-heretic-fl2va-gpustack-{backend,model}.json`。初始副本 0；指定一张已有预算的 GPU，claim 32 GiB、allocator 0.30、SHM 16 GiB，启动一个副本。FL2VA 和 T2VA 共用该实例、并发 1，与 Ref2VA 实例分开。

API 与 [原版首尾帧／纯文本 API](h3-fl2va.md) 一致，仅替换路由 URL。FL2VA 上传 `input_references` 并指定 `frame_indices`；T2VA 不上传图片，使用 `extra_params={"task":"t2va","aspect_ratio":"16:9","audio_flow_shift":3.0}`。两者均保留 `flow_shift=12` 和 `num_inference_steps=50`。生成验收使用 `scripts/check_h3_fl2va.py`，T2VA 加 `--task t2va`。

单张 SM120 GPU 已通过 5 秒 480P T2VA（168.360 秒，内网匿名）和 FL2VA（168.367 秒，内网向公网 HTTPS 带 key 调用）。两个样片均为 124 帧、24 FPS、32 kHz 双声道，完整解码与抽帧检查通过，FL2VA 首尾画面与输入对应。独立外网无凭据测试确认 401/405/404；本机错误 key 为 401、无关模型权限为 403。共卡总显存峰值 54,708 MiB，主机最少可用 436.10 GiB；既有模型容器未重启，三次 LCA 请求成功。

## 验收

已通过单张 RTX PRO 6000 Blackwell 上的 5 秒参考图生成：480P 经内网匿名路由约 37.4 秒；768P 经公网 HTTPS 地址约 60.2 秒。两者均为 124 帧、24 FPS、32 kHz 双声道，完整媒体解码通过；抽帧检查主体和场景正常。与原版同图、同提示词、同 seed 样片对比有细节差异，不据单个样片判定整体质量提升。

带 key 的 HTTPS 生成由内网发起。独立外网无凭据测试确认 TLS 可访问，无 key 为 401、错误方法为 405、非生成路径为 404；本机另外验证错误 key 为 401、H3 key 调用无关模型为 403。
