#FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
FROM alpine:latest

ARG DEBIAN_FRONTEND=noninteractive

ENV VCPKG_FORCE_SYSTEM_BINARIES="1"

#Install cmake
RUN apk update && apk add build-base cmake git zip ninja curl pkgconfig
RUN git clone https://github.com/microsoft/vcpkg.git && cd vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg integrate install
WORKDIR /vcpkg
RUN ./vcpkg install cppzmq[draft] zeromq[draft]
RUN ./vcpkg install gtest
RUN ./vcpkg install nlohmann-json