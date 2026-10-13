FROM python:3.12-slim-bookworm@sha256:2ed6491b93cd49272ee6de2b5a38440c3448360322c089fc23e370722d74179d
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY uv /usr/local/bin/uv
COPY tools /opt/lca/tools
RUN uv sync --project /opt/lca/tools/environments/nemotron-inference --frozen --no-dev --no-cache
ENV PATH=/opt/lca/tools/environments/nemotron-inference/.venv/bin:$PATH \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY server.py /opt/lca/server.py
ENTRYPOINT ["python", "/opt/lca/server.py"]
