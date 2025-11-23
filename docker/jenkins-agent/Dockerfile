FROM jenkins/inbound-agent:3345.v03dee9b_f88fc-6

USER root

# Install Python 3.10 from source and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    wget \
    lsb-release \
    gnupg \
    ca-certificates \
    apt-transport-https \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libexpat1-dev \
    libmpdec-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from source
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
    && tar xzf Python-3.10.12.tgz \
    && cd Python-3.10.12 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j$(nproc) \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip \
    && cd / && rm -rf /tmp/Python-3.10.12*

# Jenkins agent jar is already included in the base image

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Create virtual environment and install packages step by step
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Install core packages first
RUN echo "🔧 Installing core packages..." \
    && /opt/venv/bin/pip install --no-cache-dir \
        "numpy>=1.21.0" \
        "pandas>=2.0.0" \
        "scikit-learn>=1.0.0" \
        "requests>=2.28.0" \
        "flask" \
        "gunicorn" \
        "scipy" \
        "matplotlib" \
    && echo "✅ Core packages installed"

# Install ML packages
RUN echo "🔧 Installing ML packages..." \
    && /opt/venv/bin/pip install --no-cache-dir \
        "tensorflow>=2.14.0,<2.16.0" \
        "mlflow>=2.9.0" \
    && echo "✅ ML packages installed"

# Install TensorFlow Federated with fallback
RUN echo "🔧 Installing TensorFlow Federated..." \
    && (/opt/venv/bin/pip install --no-cache-dir "tensorflow-federated>=0.70.0" \
        && echo "✅ TensorFlow Federated installed") \
    || (echo "⚠️ TFF installation failed, installing minimal dependencies..." \
        && /opt/venv/bin/pip install --no-cache-dir \
            "attrs" \
            "absl-py" \
            "dm-tree" \
            "cachetools" \
        && echo "✅ TFF dependencies installed")

# Install remaining packages
RUN echo "🔧 Installing remaining packages..." \
    && /opt/venv/bin/pip install --no-cache-dir \
        "pytest>=7.0.0" \
        "pytest-cov>=4.0.0" \
        "locust>=2.0.0" \
        "prometheus-flask-exporter" \
    && echo "✅ All packages installed"
        
# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Switch back to jenkins user (base image default)
USER jenkins