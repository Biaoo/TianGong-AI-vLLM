# GPUStack CDI worker试点

本入口独立于原生模型部署器，只管理一台主机上的GPUStack worker。适用于已验证Docker原生NVIDIA CDI的Linux主机；Snap Docker/NVIDIA runtime主机不能直接套用。worker具备Docker控制权限，使用官方worker的privileged与host network方式；控制面不因此增加Docker权限。

## 配置与启停

在GPUStack创建Docker cluster，使用该cluster生成的注册token。在当前主机创建`.secrets/gpustack-worker/worker.env`，目录权限700、文件权限600，包含以下键，不提交真实值：

```dotenv
GPUSTACK_TOKEN=replace-with-cluster-registration-token
GPUSTACK_SERVER_URL=http://controller-lan-ip:17880
GPUSTACK_WORKER_NAME=chosen-worker-name
GPUSTACK_WORKER_IP=worker-lan-ip
GPUSTACK_MODEL_DIR=/absolute/existing/model/directory
```

注册token不得放在命令行、聊天或展开后的Compose输出里。私有文件用于Compose插值，不用shell source。确认模型目录已存在、权重已按清单核验，CDI目录是当前Docker实际使用的`/var/run/cdi`；worker和派生模型沿用该目录，禁用GPUStack重新生成CDI配置。

```bash
bash deploy/gpustack/worker.sh config
bash deploy/gpustack/worker.sh pull
bash deploy/gpustack/worker.sh start
bash deploy/gpustack/worker.sh status
```

控制面与worker使用同一镜像。如果管理机已有固定版本，优先从管理机`docker image save`，经已校验主机密钥的SSH传到worker执行`docker image load`，然后比较两端实际image ID及固定摘要引用。离线导入成功后跳过pull，直接config/start；不要为同一镜像重复访问公网下载。

镜像固定v2.2.3及摘要。start不拉取、不构建、不重建已有worker；配置变化使用restart。stop保留持久卷。错误详情只留私有目录，不能打印完整错误日志或Docker inspect。先确认10150、10151与推理端口范围无冲突、控制面和worker双向可达，再确认UI中Ready及GPU型号/数量。

## 模型托管

注册worker不会接管旧Compose容器。首次在明确分配给试点的GPU上创建独立deployment；保留原实例停止状态和恢复资料。选择Local Path与固定自定义后端，核对镜像实际ID、权重、entrypoint、启动参数及环境设置。镜像标签相同不代表内容相同；本地镜像需验证GPUStack不会换成远端同名镜像。

Qwen保持四卡一个副本、TP4+EP、262144上下文、GPU PLE、FlashInfer自动调优、MTP 3与NCCL。自定义entrypoint为`/bin/bash /opt/tiangong/serve.sh`时，执行命令只写模型路径及参数，不重复附加`vllm serve`。worker使用16GiB共享内存，实际模型容器也需检查。不要把GPUStack副本数设为4再给每副本4卡。

后端API字段模板见`deploy/gpustack/qwen38-backend.example.json`。这是后端定义，不包含worker/GPU身份或部署地址；创建deployment时须显式填写单worker选择器、四卡选择器、Local Path、replicas=1及backend_parameters，不能假定后端默认参数在直接调用API时自动复制。

先验证设备映射、模型运行、文本/角色/SSE/工具/图片/thinking与重启恢复，再建专用推理API key及路由验证网关。backend直连验收与网关验收分开执行；网关不一定提供`/health`等backend接口。API key与合成输出证据保存在私有目录。多个副本的负载分配和故障切换需要另外实测，一个试点副本通过不代表高可用。

## 回退

先在GPUStack停止对应deployment，确认GPU与端口释放，再恢复原生实例。禁止两套部署器同时启动同一实例。仅停止worker或控制面不能保证模型子容器已停；核对实际容器状态。保留旧镜像、权重与缓存，不执行全局prune。
