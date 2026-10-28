ARG H3_BASE_IMAGE=tiangong-h3:ee37ec06-vllm032-r6
FROM ${H3_BASE_IMAGE}
COPY deploy/h3-heretic/serve.py deploy/h3-heretic/quality.yaml deploy/h3-heretic/release.json /opt/h3-heretic/
COPY scripts/prepare_h3_heretic.py /opt/h3-heretic/
COPY deploy/h3-heretic/patch_cpu_init.py /opt/h3-heretic/
RUN python3 /opt/h3-heretic/patch_cpu_init.py
ENTRYPOINT ["python3", "/opt/h3-heretic/serve.py"]
