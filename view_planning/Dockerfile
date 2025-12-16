# ============================================================
# Stage 0 — Use pre-built dependencies (VTK + Open3D + toolchain)
# ============================================================
FROM visioncraft:deps-fixed-python AS pyenv


ENV DEBIAN_FRONTEND=noninteractive

# Core dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-venv python3-pip \
    libpython3.8 libpython3.8-dev \
    libboost-all-dev libeigen3-dev libtbb-dev \
    libjpeg-dev libpng-dev libassimp-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libarmadillo-dev liboctomap-dev libmlpack-dev \
    libssl-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libcurl4-openssl-dev pkg-config \
    libcgal-dev libcgal-qt5-dev && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# Stage 1 — Create identical virtual environment
# ============================================================
RUN mkdir -p /home/abdulaziz/playground/open3d && \
    python3.8 -m venv /home/abdulaziz/playground/open3d/env
ENV VENV=/home/abdulaziz/playground/open3d/env
RUN ${VENV}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && ${VENV}/bin/pip install --no-cache-dir pybind11==2.11.1 numpy==1.24.3 open3d==0.18.0


# ============================================================
# Stage 2 — Build Visioncraft project (isolated)
# ============================================================
FROM pyenv AS build-project
WORKDIR /app

# Copy repository safely (host remains untouched)
COPY . /app

# Clean old caches (only inside container)
RUN rm -rf build CMakeCache.txt CMakeFiles

# Environment variables
ENV PYTHON_EXECUTABLE=/home/abdulaziz/playground/open3d/env/bin/python
ENV PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
ENV PYTHON_INCLUDE_DIR=/usr/include/python3.8
ENV VTK_DIR=/usr/local/lib/cmake/vtk-9.1
ENV CMAKE_PREFIX_PATH="/usr/local/lib/cmake/Open3D"
ENV CGAL_DIR=/usr/lib/x86_64-linux-gnu/cmake/CGAL
ENV VTK_IGNORE_PYTHON_COMPONENT=ON

# Configure & build
RUN mkdir build && \
    cmake -G Ninja -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
      -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \
      -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
      -DVTK_DIR=${VTK_DIR} \
      -DCGAL_DIR=${CGAL_DIR} \
      -DVTK_IGNORE_PYTHON_COMPONENT=ON && \
    ninja -C build


# ============================================================
# Stage 3 — Runtime container
# ============================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu18.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 libboost-all-dev libeigen3-dev libtbb-dev \
    libjpeg-dev libpng-dev libassimp-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libarmadillo-dev liboctomap-dev libmlpack-dev \
    libssl-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libcurl4-openssl-dev libcgal-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build-project /usr/local /usr/local
COPY --from=build-project /home/abdulaziz/playground/open3d /home/abdulaziz/playground/open3d
COPY --from=build-project /app/build /usr/local/visioncraft

ENV PATH=/home/abdulaziz/playground/open3d/env/bin:$PATH
ENV PYTHONPATH=/usr/local/visioncraft:$PYTHONPATH
WORKDIR /usr/local/visioncraft

RUN echo '---------------------------------------------------' && \
    echo 'Visioncraft runtime environment ready.' && \
    cmake --version && \
    which python && \
    python -V && \
    echo '---------------------------------------------------'

CMD ["/bin/bash"]
