# Fixed official CUDA 13 image includes matching FlashInfer cubin, JIT shim,
# and architecture providers. Keep them together for offline inference.
FROM docker.m.daocloud.io/vllm/vllm-omni@sha256:b8a3e3c33c67f7650ef2770dcfdbd6b55da8dcf38b34801931771a603741c78e
ARG PACKAGE_INDEX=https://mirrors.aliyun.com/pypi/simple
ENV VLLM_OMNI_TARGET_DEVICE=cuda \
    VLLM_OMNI_VERSION_OVERRIDE=0.0.0+gee37ec062 \
    UV_HTTP_TIMEOUT=120
COPY upstream /opt/h3/upstream
COPY runtime-constraints.txt /opt/h3/runtime-constraints.txt
# The base image forces NCCL 2.30.7 through UV_OVERRIDE, even over explicit
# requirements. Remove only that override in this new image and honor Torch.
# The orphan OTel common package in the base requires a newer SDK than its
# other exporters and LMCache support; the installed 1.40 exporters use proto-common.
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i '/^nvidia-nccl-cu13==2.30.7$/d' /etc/uv-overrides.txt && \
    uv pip uninstall --python /usr/bin/python3 opentelemetry-exporter-otlp-common && \
    uv pip install --python /usr/bin/python3 --index-url "${PACKAGE_INDEX}" \
      --constraint /opt/h3/runtime-constraints.txt \
      "vllm==0.32.0" "nvidia-nccl-cu13==2.29.7" \
      "opentelemetry-sdk==1.40.0" /opt/h3/upstream
RUN uv pip check --python /usr/bin/python3 && \
    uv pip freeze --python /usr/bin/python3 > /opt/h3/runtime-freeze.txt
COPY serve.py h3_runtime.py quality.yaml download_hf_snapshot.py /opt/h3/
ENV PYTHONPATH=/opt/h3 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm-omni" \
      org.opencontainers.image.revision="ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f"
ENTRYPOINT ["python3", "/opt/h3/serve.py"]
