FROM alpine:3.19

# Install dependencies
RUN apk update && apk upgrade && apk add --no-cache \
    git \
    curl \
    wget \
    unzip \
    python3 \
    py3-pip \
    openjdk17-jdk \
    gnupg \
    bash \
    nss \
    ncurses \
    php

# Set up a virtual environment for Python packages
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install huggingface_hub to download datasets in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir huggingface_hub

# Create working directory
WORKDIR /app

# Create Python script using a better method to handle newlines
RUN printf '%s\n' \
    'from huggingface_hub import hf_hub_download' \
    'repo_id = "QCRI/LLMxCPG-Code"' \
    'filename = "all_source_code.zip"' \
    'print(f"Attempting to download {filename} from {repo_id}...")' \
    'file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir="/app")' \
    'print(f"Successfully downloaded to {file_path}")' > download_dataset.py

# Run the Python script to download the dataset
# Use the python command from the virtual environment
RUN python download_dataset.py

# Install Joern
RUN mkdir joern && cd joern \
    && curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o joern-install.sh \
    && chmod u+x joern-install.sh \
    # Run non-interactively (remove --interactive flag)
    && bash ./joern-install.sh --without-plugins

# Extract the downloaded zip file into Joern installation directory
RUN mkdir -p /opt/joern/joern-cli && \
    unzip -j /app/all_source_code.zip -d /opt/joern/joern-cli

# Expose the port that Joern will use
EXPOSE 9685

# Set the working directory to Joern directory
WORKDIR /opt/joern/joern-cli

CMD ["./joern", "--server", "--server-host", "0.0.0.0", "--server-port", "9685"]