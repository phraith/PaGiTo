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
### Actual building:
- ./build.ps1 on Windows
- ./build.sh on Linux
- check CUDA-version in CMakeLists.txt and set it to the correct one for the used GPU
## How to install the Python-Lib(while in folder GISAXS-ModelFitter/modules/ModelComposer)
### Needed libraries to build wheel from setup.py:
- pip install wheel
- pip install setuptools
### Build python wheel:
- python setup.py bdist_wheel
### Install created wheel:
- pip install .\dist\"wheel_name".whl
### Check:
- run one of the examples in ModelComposer/examples (be aware that the corresponding C++-server must be running!)
