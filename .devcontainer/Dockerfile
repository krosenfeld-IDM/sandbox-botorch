FROM ghcr.io/mamba-org/micromamba

RUN micromamba create -n botorch-env python=3.10 -c conda-forge -y
SHELL ["mamba", "run", "-n", "botorch-env", "/bin/bash", "-c"]

RUN micromamba install botorch -c pytorch -c gpytorch -c conda-forge -y

SHELL ["/bin/bash", "-c"]
