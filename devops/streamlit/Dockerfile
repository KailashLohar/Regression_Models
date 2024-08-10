# Base image
FROM condaforge/miniforge3:latest

# Set Label
LABEL maintainer="Kailash Lohar <kailashlohariitkgp@gmail.com>"

# Set environment variables to avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set Working Directory
WORKDIR /home

# Install Utilities with --no-install-recommends
RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends tzdata make cmake wget unzip curl git procps python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && apt-get clean
COPY streamlit.yaml .
RUN conda env create -f streamlit.yaml && conda clean --all --yes

# Set the shell to use the conda environment
SHELL ["conda", "run", "-n", "streamlit_env", "/bin/bash", "-c"]

# Expose Streamlit Port
EXPOSE 8501

# Set Entry Point
ENTRYPOINT ["conda", "run", "-n", "streamlit_env", "streamlit", "run", "main.py", "--server.port", "8501"]
