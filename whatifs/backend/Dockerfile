FROM conda/miniconda3:latest

# Building uvloop requries the C compiler in build-essential
RUN apt-get update \
    && apt-get install -yq --no-install-recommends build-essential \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    ariadne \
    uvicorn[standard] \
    starlette \
    pydgraph

COPY . /backend/
WORKDIR /backend

CMD uvicorn server:app
