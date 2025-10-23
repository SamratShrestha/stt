FROM rocm/dev-ubuntu-24.04:6.4.4-complete

ENV GFX_ARCH=gfx1151
ENV ROCM_VERSION=6.4
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV HIP_VISIBLE_DEVICES=0
ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH=gfx1151


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

#RUN wget https://raw.githubusercontent.com/wiki/ROCm/pytorch/files/install_kdb_files_for_pytorch_wheels.sh\
#    && chmod +x install_kdb_files_for_pytorch_wheels.sh \
#    && ./install_kdb_files_for_pytorch_wheels.sh \
#    && rm ./install_kdb_files_for_pytorch_wheels.sh

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

RUN python3.11 -m pip install --break-system-packages torch~=2.8.0 torchvision torchaudio~=2.2.0 --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall
RUN python3.11 -m pip install --break-system-packages transformers pandas nltk "pyannote.audio>=3.3.2,<4.0.0" faster-whisper>=1.1.1 omegaconf -U
RUN python3.11 -m pip install whisperx --break-system-packages --no-deps 

RUN python3.11 -m pip install "numpy<2.0" --break-system-packages
RUN python3.11 -m pip install matplotlib --break-system-packages
RUN python3.11 -m pip install fastapi[standard] pydantic-settings uvicorn

WORKDIR /app/
COPY ./app ./app/
