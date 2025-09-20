FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV EXECUTE_IN_DOCKER=1

# Create user and directories
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm \
    && mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

# Install system dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Upgrade pip
RUN python -m pip install --user -U pip

# Copy files
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm model_weights /opt/algorithm/model_weights
COPY --chown=algorithm:algorithm models /opt/algorithm/models
#COPY --chown=algorithm:algorithm input /input
#COPY --chown=algorithm:algorithm output /output




# Install Python dependencies
RUN python -m pip install --user -r requirements.txt

# Default entrypoint
ENTRYPOINT ["python", "-m", "inference"]
