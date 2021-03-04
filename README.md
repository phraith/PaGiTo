# GISAXS-ModelFitter

![cmake_workflow](https://github.com/phraith/GISAXS-SimFit/actions/workflows/cmake.yml/badge.svg)

## Requirements to build the C++-Framework:
### Building was tested with gcc 8.3 and MSVC 2019
### CMake (at least version 3.18):
- Link to download installer (https://cmake.org/download/)
### Eigen:
- Link to source (http://eigen.tuxfamily.org/index.php?title=Main_Page)
- Build and install with CMake
- Set EIGEN3_ROOT environment variable to path: "install_dir\"
### TBB-2020.3
- Link to source (https://github.com/oneapi-src/oneTBB/releases/tag/v2020.3)
- Can either be built with cmake (source.zip) or just copied (tbb-2020.3-x)
- Set TBB_ROOT environment variable to path: "install_dir\tbb"
### Boost
- Link to source (https://github.com/boostorg/boost/releases/tag/boost-1.74.0)
- Build and install with CMake
- Set BOOST_ROOT environment variable to path: "install_dir\"
## Requirements to use the Python-Framework (ModelComposer)
### Capn Proto:
- pip install --user pycapnp
### ZMQ:
- pip install --user zmq
