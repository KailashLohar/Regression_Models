# Base Image
FROM --platform=linux/arm64/v8 ubuntu:22.04

# Set Label
LABEL maintainer="Kailash Lohar <kailashlohariitkgp@gmail.com>"

# Set Working Directory
WORKDIR /home

# Install Utilities & Miniconda Conda Environment
RUN apt-get update && apt-get install -y make cmake wget unzip curl git gcc g++ python3 python3-pip \
    libboost-system-dev libboost-thread-dev libboost-serialization-dev libboost-filesystem-dev libboost-program-options-dev libboost-timer-dev libx11-dev libxrender-dev \
    pkg-config libhdf5-dev && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /miniconda && rm miniconda.sh && rm -rf /var/lib/apt/lists/*
ENV PATH /miniconda/bin:$PATH
RUN conda init bash
COPY environment.yaml .
RUN conda env create -f environment.yaml && conda clean --all --yes
   
# Set IOPub message rate limit
ENV JUPYTER_IOPUB_MSG_RATE_LIMIT=10000

WORKDIR /home

# Expose Port for JupyterLab
EXPOSE 7777

# Set Default Command for starting JupyterLab
CMD ["/bin/bash", "-c", "source /miniconda/etc/profile.d/conda.sh && conda activate classification && jupyter lab --ip=0.0.0.0 --port=7777 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
