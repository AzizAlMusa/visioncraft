# Base image with CUDA support for Ubuntu 18.04
FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install basic build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    software-properties-common \
    lsb-release \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Add repository for newer CMake (3.20+)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.6 and development packages
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3.6-venv \
    libpython3.6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.6 -m venv /home/abdulaziz/playground/open3d/env

# Activate virtual environment and install Python packages
RUN /home/abdulaziz/playground/open3d/env/bin/pip install --upgrade pip && \
    /home/abdulaziz/playground/open3d/env/bin/pip install \
    numpy \
    pybind11 \
    wheel \
    setuptools

# Install system dependencies for various libraries
RUN apt-get update && apt-get install -y \
    # For CGAL
    libcgal-dev \
    libcgal-qt5-dev \
    # For Boost
    libboost-all-dev \
    # For Octomap
    liboctomap-dev \
    octomap-tools \
    # For Armadillo and mlpack
    libarmadillo-dev \
    libmlpack-dev \
    # For Open3D dependencies
    libeigen3-dev \
    libflann-dev \
    libqhull-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libglfw3-dev \
    libglew-dev \
    libassimp-dev \
    # For VTK dependencies
    libxt-dev \
    libxmu-dev \
    libxi-dev \
    libx11-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libfreetype6-dev \
    # Additional utilities
    pkg-config \
    # Additional dependencies that might be needed
    libxml2-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libjsoncpp-dev \
    libexpat1-dev \
    libtbb-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Embree separately (not available in Ubuntu 18.04 default repos)
# We'll build it from source or use Intel's repository
RUN wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
    echo "deb https://apt.repos.intel.com/embree/ bionic main" > /etc/apt/sources.list.d/embree.list && \
    apt-get update && \
    apt-get install -y embree-devel || \
    (cd /tmp && \
     wget https://github.com/embree/embree/releases/download/v3.13.0/embree-3.13.0.x86_64.linux.tar.gz && \
     tar -xzf embree-3.13.0.x86_64.linux.tar.gz && \
     cp -r embree-3.13.0.x86_64.linux/* /usr/local/ && \
     rm -rf embree-3.13.0*) && \
    rm -rf /var/lib/apt/lists/*

# Build and install VTK 9.1 from source
RUN mkdir -p ~/programs && cd ~/programs && \
    wget https://www.vtk.org/files/release/9.1/VTK-9.1.0.tar.gz && \
    tar -xzf VTK-9.1.0.tar.gz && \
    mkdir -p vtk9.1/build && \
    cd vtk9.1/build && \
    cmake ../../VTK-9.1.0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=~/programs/vtk9.1/install \
        -DVTK_PYTHON_VERSION=3 \
        -DVTK_WRAP_PYTHON=ON \
        -DPYTHON_EXECUTABLE=/home/abdulaziz/playground/open3d/env/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
        -DBUILD_SHARED_LIBS=ON \
        -DVTK_USE_SYSTEM_FREETYPE=ON \
        -DVTK_USE_SYSTEM_LIBXML2=ON \
        -DVTK_USE_SYSTEM_PNG=ON \
        -DVTK_USE_SYSTEM_JPEG=ON \
        -DVTK_USE_SYSTEM_TIFF=ON \
        -DVTK_USE_SYSTEM_ZLIB=ON && \
    make -j$(nproc) && \
    make install && \
    cd ~ && \
    rm -rf ~/programs/VTK-9.1.0* ~/programs/vtk9.1/build

# Build and install Open3D from source (without OpenGL visualization to match your setup)
RUN cd /tmp && \
    git clone --depth 1 --branch v0.13.0 https://github.com/isl-org/Open3D.git && \
    cd Open3D && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CUDA_MODULE=ON \
        -DBUILD_GUI=OFF \
        -DBUILD_WEBRTC=ON \
        -DPYTHON_EXECUTABLE=/home/abdulaziz/playground/open3d/env/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_FLANN=ON \
        -DUSE_SYSTEM_JPEG=ON \
        -DUSE_SYSTEM_PNG=ON \
        -DUSE_SYSTEM_QHULL=ON \
        -DUSE_SYSTEM_GLFW=ON \
        -DUSE_SYSTEM_GLEW=ON \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF || \
    (echo "Open3D build failed, trying with embedded dependencies" && \
     cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_GUI=OFF \
        -DPYTHON_EXECUTABLE=/home/abdulaziz/playground/open3d/env/bin/python \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF) && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/Open3D

# Install additional Python packages in the virtual environment
RUN /home/abdulaziz/playground/open3d/env/bin/pip install \
    open3d==0.13.0 \
    vtk==9.1.0

# Set up CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create necessary directories for the project
RUN mkdir -p /workspace/view_planning

# Set the working directory
WORKDIR /workspace/view_planning

# Copy your project files (you'll need to add this when building)
# COPY . .

# Create a script to set up the environment and build the project
RUN echo '#!/bin/bash\n\
export PYTHON_EXECUTABLE="/home/abdulaziz/playground/open3d/env/bin/python"\n\
export PYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.6m.so"\n\
export PYTHON_INCLUDE_DIR="/usr/include/python3.6m"\n\
export VTK_DIR="~/programs/vtk9.1/install/lib/cmake/vtk-9.1"\n\
\n\
# Build the project\n\
if [ ! -d "build" ]; then\n\
    mkdir build\n\
fi\n\
cd build\n\
cmake .. \\\n\
    -DCMAKE_BUILD_TYPE=Release \\\n\
    -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \\\n\
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \\\n\
    -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \\\n\
    -DVTK_DIR=${VTK_DIR}\n\
make -j$(nproc)\n\
' > /usr/local/bin/build_project.sh && \
    chmod +x /usr/local/bin/build_project.sh

# Set up entrypoint script
RUN echo '#!/bin/bash\n\
source /home/abdulaziz/playground/open3d/env/bin/activate\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]