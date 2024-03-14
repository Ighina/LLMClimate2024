FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/lib/python3.8/site-packages/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/targets/x86_64-linux/lib/
RUN ln -s /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so.10
RUN apt-get -y update && apt-get -y install python3 && apt-get -y install curl  && apt-get -y install python3-pip
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY poetry.lock pyproject.toml /app/
# ----- Uncomment for building image for ECR -------
# COPY  configs/ /app/configs
# COPY models/ /app/models
# COPY saved_models/trained_bilstmcrf_vital-sweep-15_full /app/saved_models/trained_bilstmcrf_vital-sweep-15_full
# COPY saved_models/pretrain_08_08_2023_16_38_47_full /app/saved_models/pretrain_08_08_2023_16_38_47_full
# COPY compute_metrics.py /app/
# COPY data_utils.py /app/
# COPY alignment.py /app/
# COPY predict.py /app/
# COPY README.md /app/
# --------------------------------------------------

# Project initialization:
ENV PATH="/root/.local/bin/:$PATH"
RUN pip wheel --use-pep517 "antlr4-python3-runtime (==4.11.1)"
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root
RUN poetry install --no-interaction --no-ansi
ENV TOKENIZERS_PARALLELISM=True
