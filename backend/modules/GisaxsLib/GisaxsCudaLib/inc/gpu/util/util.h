#ifndef GISAXS_LIB_UTIL_CUH
#define GISAXS_LIB_UTIL_CUH

#include "cuda_runtime.h"

    void SumReduce(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream);
    void CalculateMaximumIntensity(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream);
    void ScaledDiffSum(float* data_real, float* data_sim, int size, float* partial_sums, float* res, float scale, cudaStream_t work_stream);
    void MultSumReduce(float* left_arr, float* right_arr, int size, float* partial_sums, float* res, cudaStream_t work_stream);
    void Preprocess(float* input, int size, float* output, float *my_max, cudaStream_t work_stream);
    void Reorder(float* input, int size, float* output, int width, int height, int blockSize, cudaStream_t work_stream);
    void Normalize(float* input, int size, unsigned char* output, cudaStream_t work_stream);

#endif