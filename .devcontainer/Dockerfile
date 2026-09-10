FROM node:20-bookworm
ARG USERNAME=node
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates gnupg2 sudo \
    python3 python3-pip python3-venv python3-dev \
    build-essential iptables ipset dnsutils iproute2 \
    && rm -rf /var/lib/apt/lists/*
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
RUN npm install -g @anthropic-ai/claude-code
RUN pip3 install --no-cache-dir --break-system-packages \
    ultralytics opencv-python-headless numpy pandas scipy statsmodels scikit-learn \
    streamlit streamlit-folium folium onnx onnxruntime matplotlib
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh
USER ${USERNAME}
WORKDIR /workspace
