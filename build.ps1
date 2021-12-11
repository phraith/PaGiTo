cmake -S . -B buildWin -DCMAKE_CXX_CPPLINT=cpplint -DCMAKE_INSTALL_PREFIX=C:\tmp\ -DCPPZMQ_BUILD_TESTS=OFF -DWITH_LIBSODIUM=OFF -DWITH_PERF_TOOL=OFF -DZMQ_BUILD_TESTS=OFF -DENABLE_CPACK=OFF
cmake --build .\buildWin\ --config Release --target install
cmake --build .\buildWin\ --config Debug