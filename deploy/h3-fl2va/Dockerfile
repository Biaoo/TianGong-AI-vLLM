ARG H3_BASE_IMAGE=tiangong-h3:ee37ec06-vllm032-r6
FROM ${H3_BASE_IMAGE}
COPY serve.py quality.yaml /opt/h3-fl2va/
ENTRYPOINT ["python3", "/opt/h3-fl2va/serve.py"]
