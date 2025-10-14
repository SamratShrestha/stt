FROM rocm/pytorch:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app/

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git cmake clang build-essential sudo && rm -rf /var/lib/apt/lists/*

# Build CTranslate2-rocm
RUN conda init bash && bash -c "conda activate py_3.9 && git clone https://github.com/arlo-phoenix/CTranslate2-rocm.git --recurse-submodules && cd CTranslate2-rocm && export CLANG_CMAKE_CXX_COMPILER=clang++ CXX=clang++ HIPCXX=\"\$(hipconfig -l)/clang\" HIP_PATH=\"\$(hipconfig -R)\" && cmake -S . -B build -DWITH_MKL=OFF -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DBUILD_TESTS=ON -DWITH_CUDNN=ON && cmake --build build -- -j16 && cd build && cmake --install . --prefix \$CONDA_PREFIX && sudo ldconfig"

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 --force-reinstall
RUN pip3 install transformers pandas nltk pyannote.audio==3.1.1 faster-whisper==1.0.1 -U
RUN pip3 install whisperx --no-deps

ENV LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/

# Install app dependencies
COPY pyproject.toml /app/
RUN pip install -e .

ENV PYTHONPATH=/app

COPY ./app /app/app
