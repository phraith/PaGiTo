FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS build

ENV VCPKG_FORCE_SYSTEM_BINARIES="1"

WORKDIR /src

# RUN apt update && apt upgrade -y
# RUN apt install cmake -y
# RUN apt-get -y install curl zip git pkg-config

RUN apt-get -y update && apt-get -y install openssh-server gcc cmake git zip curl ninja-build pkg-config

RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
WORKDIR /opt/vcpkg
RUN ./bootstrap-vcpkg.sh && ./vcpkg integrate install
RUN ./vcpkg update
#RUN apt-get -y install linux-headers
RUN ./vcpkg install pagmo2
RUN ./vcpkg install cppzmq[draft] zeromq[draft]
RUN ./vcpkg install gtest
RUN ./vcpkg install nlohmann-json
RUN ./vcpkg install spdlog

RUN apt-get -y install  gdb

WORKDIR /root

RUN echo 'root:docker' | chpasswd

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

COPY ./docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

#docker run --gpus all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -p5555:5555 -p2222:22 --name gisaxs_remote_dev cuda_dev:0.3