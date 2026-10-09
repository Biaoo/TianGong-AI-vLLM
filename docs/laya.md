# Laya Multilingual Docker 部署

## 运行拓扑

Laya 使用专用 SDK 和 PyTorch，不通过 vLLM serve 加载自定义决策头。`deploy/profiles/laya-multilingual.json` 选择通用 `replicated-http` 运行方式：每个 GPU 对应一个独立 worker 容器，每个容器只看到一张 GPU、持有完整模型；Nginx 在统一入口按最少连接数分发请求。不使用张量并行、跨卡拆分或 NCCL 通信。

```text
客户端 :7720 ─ Nginx ─ worker-0 ─ GPU 0 / 完整 Laya 模型
                    ├ worker-1 ─ GPU 1 / 完整 Laya 模型
                    └ worker-2 ─ GPU 2 / 完整 Laya 模型
```

改变本机实例的 gpus 列表即可选择单卡或多卡。实例镜像、模型路径、上下文和端口仍由模型配置与机器配置两层管理，不影响其他模型实例。worker 不发布宿主端口；网关启动前等待所有 worker 的健康检查。运行期间 Nginx 使用 Docker DNS 更新地址，并对连接错误提供被动故障转移；网关健康代表至少一个后端可用，全部副本是否可用由实例 status 和实际请求分布共同验证。

## 固定版本与准备

- Laya SDK 0.3.4，wheel SHA256 固定在 `deploy/laya/requirements.txt`。
- 复用固定摘要的 vLLM 0.29.0 镜像提供 Torch 2.13.0+cu130 / CUDA 13.0 与 Transformers 5.16.1；容器入口运行 Laya 专用 HTTP 服务，不启动 vLLM。
- Nginx 1.31.1，镜像摘要固定在模型配置。
- ModelScope `convaiinnovations/laya-multilingual` revision `5ad6e84d70d70edde0d5a3241c5233d0b11cd6b8`；六个运行文件按清单下载并校验 SHA256。

模型目录只读挂载。SDK 会修复 tokenizer 元数据格式，因此启动时创建临时目录，仅复制可变的 tokenizer_config.json，其余文件链接到只读模型；原始已验证文件保持不变。USE_TF=0 避免探测 TensorFlow；启动及请求完成后检查实际 CUDA 驻留，SDK 回退 CPU 会使实例验收失败。

```bash
./deploy/manage.sh init laya-single laya
# 编辑 deployments/laya.json，核对 GPU、端口和模型目录
./deploy/manage.sh build laya
./deploy/manage.sh download laya
./deploy/manage.sh start laya
./deploy/manage.sh check laya
python3 scripts/check_laya.py --instance laya --benchmark --label single-gpu
```

单卡通过后，把该实例 gpus 改为实际三张 GPU 的列表（模板见 `deploy/examples/laya.json`），执行 restart 或 restart-loaded，再运行相同验收与基准。start 不会重建已有网关，因此扩缩容属于配置变化，必须 restart。restart 会移除本实例项目中不再声明的 worker，避免缩容后旧副本继续占用 GPU。

## HTTP API

默认入口 `http://127.0.0.1:7720`。模型名 `convaiinnovations/laya-multilingual`；POST `/v1/decisions` 使用 SDK 的 state/questions 结构：

```json
{
  "model": "convaiinnovations/laya-multilingual",
  "state": "这笔订单被重复扣款了，请财务部门今天把多收的钱退给我。",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "payments, invoices, refunds",
        "technical": "software bugs and outages",
        "sales": "pricing and new purchases"
      }
    },
    "refund": {"type": "noul", "instructions": "Does the sender request a refund?"}
  }
}
```

支持 choice、score、noul 三种类型；响应包含 answers 和 usage，`X-Model-Replica` 响应头标识处理副本，用于检查流量分布。GET `/v1/models` 返回模型身份及上下文；`/health` 检查 CUDA 驻留；`/metrics` 返回该副本的在途、排队、完成、失败请求数及 CUDA 分配/保留/峰值内存。网关上的 metrics 仅代表本次选中的一个副本，完整监测需在各 worker 上分别读取。

默认无鉴权；实例 api_key_env 指向 LAYA_API_KEY，设置非空变量并重启后业务请求和 metrics 需要 Bearer，health 无需鉴权。默认最大 32 个问题/请求，choice/score 最多 20 个选项；请求体最大 1 MiB，state 最大 128 KiB。每个问题按 SDK 规则截断至默认 1024 tokens，其中问题/选项最多占 256 tokens；长文本可能被截断，不能将该接口当作全文处理接口。

每个副本仅允许一次 GPU 推理执行，默认最多接纳 64 个执行中或等待中的请求，超过容量返回 429；排队超过 30 秒返回 503。无效请求返回 400，未知模型 404。取消 HTTP 请求不会提前释放仍在推理的 GPU 槽位。容器停止宽限 120 秒，先检查各 worker 请求已排空。

## 验收与性能

`manage.sh check laya` 验证中文退款样本、英文和结构化输入、三种类型的概率/数值范围、无效输入拒绝及并发请求覆盖所有 GPU 副本。结果保存在私有 `output/validation/laya-<UTC时间>.json`。

`check_laya.py --benchmark` 默认三轮，每轮 90 请求、12 路并发、每请求 8 个问题，使用相同固定文本并预热。报告请求/秒、问题/秒、P50/P95 和副本分布。单卡与多卡比较必须使用相同镜像、输入、参数和并发，确认既有共享 GPU 服务没有负载变化后解释差异；这不是业务准确率或长时间容量测试。

模型原始概率未经业务校准，线上业务应另用标注数据评估；默认上下文保持检查点的 1024。仅调整底层支持的更长上下文不意味着该模型已通过长输入质量验证。

官方来源：[模型](https://modelscope.cn/models/convaiinnovations/laya-multilingual)、[SDK](https://github.com/NandhaKishorM/laya)。

## 已验证的三卡部署

2026-09-21 在三张 RTX PRO 6000 Blackwell 上先完成单卡验收，再扩展为三卡各一个完整副本。单卡与三卡使用同一最终镜像；上述默认基准每请求 8 个问题、合计 6624 输入 tokens、12 路并发、每轮 90 请求、三轮取中位数：

| 指标 | 单卡 | 三卡 |
| --- | ---: | ---: |
| 请求/秒 | 40.13 | 120.03 |
| 问题/秒 | 321.06 | 960.26 |
| P95 请求延迟（三轮范围） | 0.301–0.303 秒 | 0.100–0.103 秒 |

吞吐比约 2.99，每轮三张卡各处理 30 个请求。另在三个副本同时验证每请求 32 问题、每问题 1024 tokens（合计 32768 tokens）；均成功返回，33 问题请求被 HTTP 400 拒绝。此处 32768 是多个问题的 token 总量，不是单问题上下文上限。

普通测试后每副本进程约 2.5 GiB 显存；完成最大批量检查后，nvidia-smi 显示每副本 3978 MiB（约 3.9 GiB，含缓存和 CUDA 上下文）。Torch 峰值已分配约 2.33 GiB，保留约 3.20 GiB。三副本均 healthy，推理失败计数为零；现有 embedding 实测通过，MinerU 健康且既有模型进程未被重启。完整部署与性能证据保存在私有 output/laya-deployment.json。

上述为固定样本的短时对照，不代表真实业务持续吞吐、模型全面准确率或整机重启验收。
