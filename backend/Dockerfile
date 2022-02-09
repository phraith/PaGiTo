FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

#Install cmake
RUN apt-get -y update
RUN apt install -y openssh-server
RUN apt-get -y install vim build-essential libtool autoconf unzip wget --fix-missing
RUN apt-get -y install libssl-dev libhiredis-dev --fix-missing
RUN apt-get -y install gcc gdb
RUN apt-get -y install python3.7 python3-pip
RUN pip3 install cpplint
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.2/cmake-3.21.2.tar.gz
RUN tar -xzvf cmake-3.21.2.tar.gz
RUN cd cmake-3.21.2 && ./bootstrap && make -j$(nproc) && make install

RUN apt-get -y install curl zip git pkg-config
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
WORKDIR /opt/vcpkg
RUN ./bootstrap-vcpkg.sh && ./vcpkg integrate install && ./vcpkg integrate bash && echo 'export PATH=$PATH:/opt/vcpkg' >>~/.bashrc
RUN ./vcpkg install pagmo2
RUN ./vcpkg install cppzmq[draft] zeromq[draft]
RUN ./vcpkg install capnproto
RUN ./vcpkg install gtest
RUN ./vcpkg install nlohmann-json

RUN apt-get -y install htop

WORKDIR /root

RUN echo 'root:docker' | chpasswd

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]