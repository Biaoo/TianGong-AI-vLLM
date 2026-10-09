# Build from the verified worker image already imported over the LAN.
ARG WORKER_BASE_IMAGE=gpustack/gpustack:v2.2.3
FROM ${WORKER_BASE_IMAGE}
COPY patch_snap_pause.py /tmp/patch_snap_pause.py
RUN python3 /tmp/patch_snap_pause.py && rm /tmp/patch_snap_pause.py
