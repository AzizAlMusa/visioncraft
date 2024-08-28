# Use an Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages, including X11 libraries and CGAL
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    git \
    cmake \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    libpcl-dev \
    libvtk7-dev \
    liboctomap-dev \
    libeigen3-dev \
    x11-apps \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    clang \
    libc++-dev \
    libc++abi-dev \
    libcgal-dev \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Install CMake 3.24
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh \
    && chmod +x cmake-3.24.0-linux-x86_64.sh \
    && ./cmake-3.24.0-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.24.0-linux-x86_64.sh

# Install Open3D C++ library
RUN git clone --recursive https://github.com/intel-isl/Open3D.git \
    && cd Open3D \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_PYTHON_MODULE=ON .. \
    && make -j$(nproc) \
    && make install

# Set the working directory to /app
WORKDIR /app

# Copy the project files into the container
COPY . .

# Build the view_planning application
WORKDIR /app/view_planning
RUN rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)

# Allow X11 forwarding (adjust as needed for your host's DISPLAY environment variable)
ENV DISPLAY=$DISPLAY

# Command to run the application (you can customize this as needed)
# CMD ["./build/your_executable_name"]

# Clean up unnecessary files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
