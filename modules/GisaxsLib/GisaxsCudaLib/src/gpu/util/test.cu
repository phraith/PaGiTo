#include "gpu/util/test.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

__global__ void cuda_add(int *x, int *y, int *res)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    res[idx] = x[idx] + y[idx];
}

__global__ void cuda_assign(MyType* out, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= size)
        return;


    out[idx] = idx;
}

__global__ void cuda_schwefel(MyType* out, MyType* x, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
        return;

    out[idx] = x[idx] * sinf(sqrtf(abs(x[idx])));
}

__global__ void cuda_img_diff(MyType* out_img, MyType* real_img, MyType val, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
        return;

    out_img[idx] = fabsf(real_img[idx] - val);
}

extern "C++"
{
    void add(int *x, int *y, int *res, int elems)
    {
        cuda_add<<<1, 512>>>(x,y,res);
    }
    void assign_value(MyType* out, int size, cudaStream_t work_stream)
    {
        cuda_assign <<< (size / 512) + 1, 512, 0, work_stream>>>(out, size);
    }

    void schwefel(MyType* out, MyType *x, int size, cudaStream_t work_stream)
    {
        cuda_schwefel << < (size / 128) + 1, 128, 0, work_stream >> > (out, x, size);
    }

    void img_diff(MyType* in_img, MyType* real_img, MyType val, int size, cudaStream_t work_stream)
    {
        cuda_img_diff << < (size / 256) + 1, 256, 0, work_stream >> > (in_img, real_img, val, size);
    }
}