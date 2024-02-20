FROM ghcr.io/mamba-org/micromamba

# Install Git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
    
RUN micromamba create -n botorch-env python=3.10 -c conda-forge -y
SHELL ["micromamba", "run", "-n", "botorch-env", "/bin/bash", "-c"]

RUN micromamba install botorch jupyterlab -c pytorch -c gpytorch -c conda-forge -y

SHELL ["/bin/bash", "-c"]
