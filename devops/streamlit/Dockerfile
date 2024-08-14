# Base image
FROM --platform=linux/arm64/v8 ubuntu:22.04

# Set Label
LABEL maintainer="Kailash Lohar <kailashlohariitkgp@gmail.com>"

# Set Working Directory
WORKDIR /home

# Install Utilities
RUN apt-get update && apt-get install -y wget curl git python3 python3-pip && \
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh && \
    bash miniconda.sh -b -p /miniconda && rm miniconda.sh && rm -rf /var/lib/apt/lists/*
ENV PATH /miniconda/bin:$PATH
RUN conda init bash
COPY streamlit.yaml .
RUN conda env create -f streamlit.yaml && conda clean --all --yes

# Expose Streamlit Port
EXPOSE 8501

# Set Entry Point
CMD ["/bin/bash", "-c", "source /miniconda/etc/profile.d/conda.sh && conda activate streamlit_env && streamlit run main.py --server.port 8501"]
