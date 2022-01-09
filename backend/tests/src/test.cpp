#include "gtest/gtest.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "gpu/util/test.h"

 TEST(CudaTests, TwoTimes) {
      int *x_dev = nullptr, *y_dev = nullptr, *res_dev = nullptr, *res_host = nullptr;

     res_host = new int[512];

     cudaMalloc((void**)&x_dev, 512 * sizeof(int));
     cudaMalloc((void**)&y_dev, 512 * sizeof(int));
     cudaMalloc((void**)&res_dev, 512 * sizeof(int));

     for (int i = 0; i < 512; ++i)
     {
         res_host[i] = i * 2;
     }

     cudaMemcpy(x_dev, res_host, 512 * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(y_dev, res_host, 512 * sizeof(int), cudaMemcpyHostToDevice);

     add(x_dev, y_dev, res_dev, 512);

     cudaMemcpy(res_host, res_dev, 512 * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(x_dev);
     cudaFree(y_dev);
     cudaFree(res_dev);

     for(int i = 0; i < 512; ++i)
     {
         EXPECT_EQ(4 * i, res_host[i]);
     }

    delete[] res_host;
 }

 TEST(CudaTests, FourTimes) {
     int* x_dev = nullptr, * y_dev = nullptr, * res_dev = nullptr, * res_host = nullptr;

     res_host = new int[512];

     cudaMalloc((void**)&x_dev, 512 * sizeof(int));
     cudaMalloc((void**)&y_dev, 512 * sizeof(int));
     cudaMalloc((void**)&res_dev, 512 * sizeof(int));

     for (int i = 0; i < 512; ++i)
     {
         res_host[i] = i * 4;
     }

     cudaMemcpy(x_dev, res_host, 512 * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(y_dev, res_host, 512 * sizeof(int), cudaMemcpyHostToDevice);

     add(x_dev, y_dev, res_dev, 512);

     cudaMemcpy(res_host, res_dev, 512 * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(x_dev);
     cudaFree(y_dev);
     cudaFree(res_dev);

     for (int i = 0; i < 512; ++i)
     {
         EXPECT_EQ(8 * i, res_host[i]);
     }

     delete[] res_host;
 }