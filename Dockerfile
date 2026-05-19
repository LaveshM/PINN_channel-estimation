FROM pytorch/pytorch:latest

WORKDIR /workspace

# system tools (optional but useful)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy==1.26.4
RUN pip install python-swiftclient python-keystoneclient

# default command
CMD ["bash"]