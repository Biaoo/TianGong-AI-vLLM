ARG H3_BASE_IMAGE=tiangong-h3-fl2va:ee37ec06-fp8-r1
FROM ${H3_BASE_IMAGE}
COPY deploy/h3-heretic-fl2va/serve.py deploy/h3-heretic-fl2va/release.json /opt/h3-heretic-fl2va/
COPY scripts/prepare_h3_heretic.py /opt/h3-heretic-fl2va/
ENTRYPOINT ["python3", "/opt/h3-heretic-fl2va/serve.py"]
