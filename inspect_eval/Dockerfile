# Dockerfile (builds an image WITH CUDA + the editor service)
FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN pip install --no-cache-dir inspect-tool-support && \
    inspect-tool-support post-install --no-web-browser

ENV PYTHONUNBUFFERED=1
CMD ["sleep", "infinity"]
