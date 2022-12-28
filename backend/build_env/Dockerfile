FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS build

# ENV VCPKG_FORCE_SYSTEM_BINARIES "1"
ENV ASAN_OPTIONS protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

WORKDIR /src

RUN apt-get -y update && apt-get -y --no-install-recommends install openssh-server gcc cmake git zip curl ninja-build pkg-config gdb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
WORKDIR /opt/vcpkg
RUN mkdir ~/.vcpkg && touch ~/.vcpkg/vcpkg.path.txt && ./bootstrap-vcpkg.sh && ./vcpkg integrate install
RUN ./vcpkg update
RUN ./vcpkg install cppzmq[draft] zeromq[draft]
RUN ./vcpkg install gtest
RUN ./vcpkg install nlohmann-json
RUN ./vcpkg install spdlog
RUN ./vcpkg install eigen3

WORKDIR /root

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

# RUN useradd -m user \
#   && yes password | passwd user

# RUN usermod -s /bin/bash user
RUN echo 'export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0' >> /etc/profile
RUN echo "root:root" | chpasswd
RUN echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> /etc/profile
RUN echo 'while read p; do   echo export "$p"; done < /etc/environment' >> /etc/profile
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

#docker run --gpus all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -p5555:5555 -p2222:22 --name gisaxs_remote_dev cuda_dev:0.3