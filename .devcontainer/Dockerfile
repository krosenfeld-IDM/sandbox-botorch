FROM ghcr.io/mamba-org/micromamba

# # Install Git
RUN sudo apt-get update && \
    sudo apt-get install -y git && \
    sudo rm -rf /var/lib/apt/lists/*
    
RUN micromamba create -n botorch-env python=3.10 -c conda-forge -y
SHELL ["micromamba", "run", "-n", "botorch-env", "/bin/bash", "-c"]

RUN micromamba install botorch jupyterlab -c pytorch -c gpytorch -c conda-forge -y

SHELL ["/bin/bash", "-c"]
