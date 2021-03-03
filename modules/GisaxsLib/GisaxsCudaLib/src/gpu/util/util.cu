#include "gpu/util/util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "stdio.h"

__global__ void reduce_sum(float* in, float* sum, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        local_sum += in[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int active_threads = blockDim.x >> 1; active_threads; active_threads >>= 1) {
        if (tid < active_threads) {
            sdata[tid] += sdata[tid + active_threads];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_mult_add(float* left_arr, float* right_arr, int n, float *sum)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        local_sum += left_arr[i] * right_arr[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int active_threads = blockDim.x >> 1; active_threads; active_threads >>= 1) {
        if (tid < active_threads) {
            sdata[tid] += sdata[tid + active_threads];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_scaled_diff(float* in_real, float* in_sim, float* sum, float scale, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;


    float local_sum = 0;
    float epsilon = 0.00000001;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {

        float real_int = in_real[i];
        float sim_int = in_sim[i] * scale;
        float abs_diff = real_int - sim_int;
        //local_sum += (abs_diff * abs_diff) / real_int;
        //printf("%f %f %f\n", in_real[i], in_sim[i], (abs_diff * abs_diff));
        local_sum += (abs_diff * abs_diff);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int active_threads = blockDim.x >> 1; active_threads; active_threads >>= 1) {
        if (tid < active_threads) {
            sdata[tid] += sdata[tid + active_threads];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_max(float* in, float* sum, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float local_max = 0;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        float val = in[i];
        if (val > local_max)
            local_max = val;
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int active_threads = blockDim.x >> 1; active_threads; active_threads >>= 1) {
        if (tid < active_threads) {
            float val = sdata[tid + active_threads];
            if (val > sdata[tid])
                sdata[tid] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

void max(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_max << <num_blocks, num_threads, shared_size, work_stream >> > (data, partial_sums, size);
    reduce_max << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}

void SumReduce(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_sum << <num_blocks, num_threads, shared_size, work_stream >> > (data, partial_sums, size);
    reduce_sum << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}

void ScaledDiffSum(float* data_real, float *data_sim, int size, float* partial_sums, float* res, float scale, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_scaled_diff << <num_blocks, num_threads, shared_size, work_stream >> > (data_real, data_sim, partial_sums, scale, size);
    reduce_sum << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}

void MultSumReduce(float* left_arr, float* right_arr, int size, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_mult_add << <num_blocks, num_threads, shared_size, work_stream >> > (left_arr, right_arr, size, partial_sums);
    reduce_sum << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}