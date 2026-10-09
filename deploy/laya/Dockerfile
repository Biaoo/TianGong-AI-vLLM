# Reuse the already validated Blackwell-compatible Torch/CUDA stack.
FROM vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
COPY deploy/laya/requirements.txt /opt/tiangong/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /opt/tiangong/requirements.txt
ENV USE_TF=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
COPY deploy/laya/server.py /opt/tiangong/laya_server.py
ENTRYPOINT ["python3", "/opt/tiangong/laya_server.py"]
