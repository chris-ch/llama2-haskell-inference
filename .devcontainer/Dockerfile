FROM debian:bookworm-slim

ARG GHCUP_DWN_URL=https://downloads.haskell.org/~ghcup/x86_64-linux-ghcup

ARG PATH
RUN test -n ${PATH}
ENV PATH=${PATH}
ENV LANG=C.UTF-8
ENV BOOTSTRAP_HASKELL_NONINTERACTIVE=1

RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl \
        libnuma-dev \
        zlib1g-dev \
        libgmp-dev \
        libgmp10 \
        git \
        wget \
        lsb-release \
        software-properties-common \
        gnupg2 \
        apt-transport-https \
        gcc \
        autoconf \
        automake \
        libffi-dev \
        libffi8 \
        libgmp-dev \
        libgmp10 \
        libncurses-dev \
        libncurses5 \
        libtinfo5 \
        libblas3 \
        liblapack3 \
        liblapack-dev \
        libblas-dev \
        xz-utils \
        build-essential

# install gpg keys
RUN \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys 7D1E8AFD1D4A16D71FADA2F2CCC85C0E40C06A8C && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys FE5AB6C91FEA597C3B31180B73EDE9E8CFBAEF01 && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys 88B57FCF7DB53B4DB3BFA4B1588764FBE22D19C4 && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys EAF2A9A722C0C96F2B431CA511AAD8CEDEE0CAEF

# install ghcup
RUN \
    curl ${GHCUP_DWN_URL} > /usr/bin/ghcup && \
    chmod +x /usr/bin/ghcup && \
    ghcup config set gpg-setting GPGStrict

ARG VERSION_GHC=9.8.1
ARG VERSION_CABAL=latest
ARG VERSION_STACK=latest

# install GHC, cabal and stack
RUN \
    ghcup -v install ghc --isolate /usr/local --force ${VERSION_GHC} && \
    ghcup -v install cabal --isolate /usr/local/bin --force ${VERSION_CABAL} && \
    ghcup -v install stack --isolate /usr/local/bin --force ${VERSION_STACK} && \
    ghcup install hls

ARG USER_NAME=haskell
RUN useradd --no-log-init --create-home --shell /bin/bash ${USER_NAME}
WORKDIR /home/${USER_NAME}

USER ${USER_NAME}
