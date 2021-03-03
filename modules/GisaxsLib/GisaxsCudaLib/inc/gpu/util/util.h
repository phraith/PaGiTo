#ifndef GISAXS_LIB_UTIL_CUH
#define GISAXS_LIB_UTIL_CUH

#include "cuda_runtime.h"

extern "C++"
{
	void SumReduce(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream);
	void max(float* data, int size, float* partial_sums, float* res, cudaStream_t work_stream);
	void ScaledDiffSum(float* data_real, float* data_sim, int size, float* partial_sums, float* res, float scale, cudaStream_t work_stream);
	void MultSumReduce(float* left_arr, float* right_arr, int size, float* partial_sums, float* res, cudaStream_t work_stream);
}

#endif