FROM rocm/dev-ubuntu-24.04:6.4.4-complete

ENV GFX_ARCH=gfx1100
ENV ROCM_VERSION=6.4
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV HIP_VISIBLE_DEVICES=0
ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH=gfx1100
ENV PYTHONUNBUFFERED=1

RUN apt update \
    && apt install -y --no-install-recommends \
    software-properties-common wget ca-certificates gnupg libjpeg-dev hipblas \
    && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt update \
    && apt install -y \
    libjpeg-dev \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    wget \
    hipblas \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m ensurepip

RUN python3.11 -m pip install --upgrade --ignore-installed pip setuptools wheel && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --set python /usr/bin/python3.11

RUN python3.11 -m pip install --break-system-packages wheel setuptools \
    && python3.11 -m pip install --no-cache-dir --break-system-packages --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4/ \
    && rm -rf ~/.cache/pip

RUN apt update \
    && apt install -y \
    git \
    cmake \
    rocm-libs \
    ffmpeg \
    wget \
    libomp5 \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/whisperx

RUN git clone https://github.com/arlo-phoenix/CTranslate2-rocm.git --recurse-submodules && \
    cd CTranslate2-rocm && \
    CLANG_CMAKE_CXX_COMPILER=amdclang++ \
    CXX=amdclang++ \
    HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DOPENMP_RUNTIME=COMP \
        -DWITH_MKL=OFF \
        -DWITH_HIP=ON \
        -DCMAKE_HIP_ARCHITECTURES=$PYTORCH_ROCM_ARCH \
        -DBUILD_TESTS=ON \
        -DWITH_CUDNN=ON && \
    cmake --build build -- -j$(nproc) && \
    cd build && \
    cmake --install . && \
    ldconfig
    
RUN cd /opt/whisperx/CTranslate2-rocm/python && \
    python3.11 -m pip install --break-system-packages --no-cache-dir -r install_requirements.txt && \
    python3.11 setup.py bdist_wheel && \
    python3.11 -m pip install --break-system-packages dist/*.whl
    
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib/

RUN python3.11 -m pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall
RUN python3.11 -m pip install --break-system-packages transformers pandas nltk pyannote.audio==3.1.1 faster-whisper==1.1.1 -U
RUN python3.11 -m pip install whisperx --break-system-packages --no-deps

RUN python3.11 -m pip install "numpy<2.0" --break-system-packages
RUN python3.11 -m pip install matplotlib --break-system-packages

WORKDIR /app/

# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Place executables in the environment at the front of the path
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/app/.venv/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/app

COPY ./pyproject.toml ./uv.lock /app/

COPY ./app /app/app

# Sync the project
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

CMD ["fastapi", "run", "app/main.py"]
