FROM ghcr.io/mamba-org/micromamba

USER root

# Install Git
# Install Git
RUN apk update && \
    apk add git
    
USER 1000

RUN micromamba create -n botorch-env python=3.10 -c conda-forge -y
SHELL ["micromamba", "run", "-n", "botorch-env", "/bin/bash", "-c"]

RUN micromamba install botorch jupyterlab -c pytorch -c gpytorch -c conda-forge -y

SHELL ["/bin/bash", "-c"]
