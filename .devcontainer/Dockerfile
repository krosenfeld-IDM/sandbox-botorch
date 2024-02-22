FROM ghcr.io/mamba-org/micromamba

USER root

# Install Git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

USER mambauser
    
RUN micromamba create -n botorch-env python=3.10 -c conda-forge -y
SHELL ["micromamba", "run", "-n", "botorch-env", "/bin/bash", "-c"]

RUN micromamba install botorch jupyterlab pytorch torchvision -c pytorch -c gpytorch -c conda-forge -y
COPY requirements.txt .
RUN pip install -r requirements.txt

SHELL ["/bin/bash", "-c"]
