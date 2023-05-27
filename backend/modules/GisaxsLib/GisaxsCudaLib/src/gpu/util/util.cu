#include "gpu/util/util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

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
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {

        float real_int = in_real[i];
        float sim_int = in_sim[i] * scale;
        float abs_diff = real_int - sim_int;
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

__global__ void calculate_diff(float* in_real, float* in_sim, float* out_diff, int n) {

    int tid = threadIdx.x;

    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        out_diff[i] = fabs(in_real[i] - in_sim[i]);
    }
}

__global__ void calculate_diff_const(float* in_data, float* const_value, float* out_diff, int n) {

    int tid = threadIdx.x;

    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        out_diff[i] = fabs(in_data[i] - *const_value);
    }
}

__device__ float3 scalar_mult(float factor, float3 vec)
{
    return { factor * vec.x, factor * vec.y, factor * vec.z };
}

__device__ float3 vector_add(float3 a, float3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__global__ void preprocess_easy(float* in, float* out, float* my_max, int n)
{
    int tid = threadIdx.x;
    float logmax = logf(*my_max);
    float logmin = logf(fmaxf(2.f, 1e-10f * *my_max));
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {

        float log_val = logf(in[i]);
        log_val -= logmin;
        log_val /= (logmax - logmin);
        out[i] = log_val;
    }
}

__global__ void normalize(float* in, unsigned char* out, int n)
{
    int tid = threadIdx.x;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        out[i] = (unsigned char)(in[i] * 255.0);
    }
}

__global__ void reorder(float* in, float* out, int size, int width, int height, int blocksize)
{
    int tid = threadIdx.x;

    int blockMaxX = width / blocksize;

    int blockCells = blocksize * blocksize;
    for (int i = blockDim.x * blockIdx.x + tid; i < size ; i += blockDim.x * gridDim.x)
    {
        int yCoord = i / width;
        int xCoord = i % width;

        int blockIdX = xCoord / blocksize;
        int blockIdY = yCoord / blocksize;

        int posBlockX = xCoord % blocksize;
        int posBlockY = yCoord % blocksize;

        int blockId = blockIdX * blockMaxX + blockIdY;
        int startPos = blockId * blockCells;
        int newIdx = startPos + (posBlockX * blocksize) + posBlockY;

        out[newIdx] = in[i];
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

__global__ void reduce_min(float* in, float* sum, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float local_min = FLT_MAX;
    for (int i = blockDim.x * blockIdx.x + tid; i < n; i += blockDim.x * gridDim.x)
    {
        float val = in[i];
        if (val < local_min)
            local_min = val;
    }

    sdata[tid] = local_min;
    __syncthreads();

    for (int active_threads = blockDim.x >> 1; active_threads; active_threads >>= 1) {
        if (tid < active_threads) {
            float val = sdata[tid + active_threads];
            if (val < sdata[tid])
                sdata[tid] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

void CalculateDiff(float* data_real, float *data_sim, int size, float* out_diff, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;
    calculate_diff<<<num_blocks, num_threads, 0, work_stream>>>(data_real, data_sim, out_diff, size);

    int shared_size = num_threads * sizeof(float);
    reduce_min << <num_blocks, num_threads, shared_size, work_stream >> > (out_diff, partial_sums, size);
    reduce_min << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);

    calculate_diff_const<<<num_blocks, num_threads, 0, work_stream>>>(out_diff, res, out_diff, size);

    reduce_sum << <num_blocks, num_threads, shared_size, work_stream >> > (out_diff, partial_sums, size);
    reduce_sum << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}


void CalculateMaximumIntensity(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_max << <num_blocks, num_threads, shared_size, work_stream >> > (data, partial_sums, size);
    reduce_max << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}

void Preprocess(float* input, int size, float* output, float *my_max, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    //preprocess << < num_blocks, num_threads, shared_size, work_stream >> > (input, output, my_max, size);
    preprocess_easy << < num_blocks, num_threads, shared_size, work_stream >> > (input, output, my_max, size);
}

void Normalize(float* input, int size, unsigned char* output, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    //preprocess << < num_blocks, num_threads, shared_size, work_stream >> > (input, output, my_max, size);
    normalize << < num_blocks, num_threads, shared_size, work_stream >> > (input, output, size);
}

void Reorder(float* input, int size, float *output, int width, int height, int blockSize, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reorder << < num_blocks, num_threads, shared_size, work_stream >> > (input, output, size, width, height, blockSize);
}

void SumReduce(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream)
{
    int num_threads = 256;
    int num_blocks = 256;

    int shared_size = num_threads * sizeof(float);

    reduce_sum << <num_blocks, num_threads, shared_size, work_stream >> > (data, partial_sums, size);
    reduce_sum << <1, num_threads, shared_size, work_stream >> > (partial_sums, res, num_blocks);
}

void MinReduce(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream)
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