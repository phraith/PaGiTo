FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04 AS base
WORKDIR /app
EXPOSE 5555

FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS build
WORKDIR /src
COPY . .

RUN apt update && apt upgrade -y
RUN apt install cmake -y
RUN apt-get -y install curl zip git pkg-config
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
WORKDIR /opt/vcpkg
RUN ./bootstrap-vcpkg.sh && ./vcpkg integrate install && ./vcpkg integrate bash && echo 'export PATH=$PATH:/opt/vcpkg' >>~/.bashrc
RUN ./vcpkg install pagmo2
RUN ./vcpkg install cppzmq[draft] zeromq[draft]
RUN ./vcpkg install capnproto
RUN ./vcpkg install gtest
RUN ./vcpkg install nlohmann-json
RUN ./vcpkg install spdlog
WORKDIR /src
RUN cmake -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -B./build -S. && cmake --build ./build

FROM base AS final
RUN apt update -y && apt install -y build-essential
WORKDIR /app
COPY --from=build /src/build .
ENTRYPOINT ["./apps/ModelingFramework/ModelingFramework"]