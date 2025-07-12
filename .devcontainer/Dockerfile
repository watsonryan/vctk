FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang lldb lld ninja-build cmake git curl wget \
    libeigen3-dev libtbb-dev \
    clang-format \
    libsqlite3-dev libaec-dev zlib1g-dev \
    ca-certificates sudo \
    zsh \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt && \
    curl -Ls https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake \
         -o /opt/CPM.cmake

# ------------ VS Code user ---------------------------------------------------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd  --uid $USER_UID --gid $USER_GID -m $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/90-vscode \
 && chmod 0440 /etc/sudoers.d/90-vscode \
 && chsh --shell /usr/bin/zsh $USERNAME

USER vscode
RUN [ ! -f ~/.zshrc ] && echo 'export PROMPT="%n@%m:%~ %# "' > ~/.zshrc || true
WORKDIR /workspaces
ENV CPM_SOURCE_CACHE=/workspaces/.cpm-cache
SHELL ["/usr/bin/zsh", "-c"]