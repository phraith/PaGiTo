#ifndef MODULES_COMMON_INC_CUDA_TYPES_STANDARD_VECTOR_TYPES_H
#define MODULES_COMMON_INC_CUDA_TYPES_STANDARD_VECTOR_TYPES_H

#include "vector_types.h"
#include <cuComplex.h>

typedef struct gpu_info_t {
    char name[256];                  /**< ASCII string identifying device */
    size_t totalGlobalMem;           /**< Total global memory available on device in bytes */
    size_t freeGlobalMem;            /**< Free global memory available on device in bytes */
    int sharedMemPerBlock;           /**< Shared memory available per block in bytes */
    int regsPerBlock;                /**< 32-bit registers available per block */
    int warpSize;                    /**< Warp size in threads */
    int maxThreadsPerBlock;          /**< Maximum number of threads per block */
    int maxThreadsDim[3];            /**< Maximum size of each dimension of a block */
    int maxGridSize[3];              /**< Maximum size of each dimension of a grid */
    int totalConstMem;               /**< Constant memory available on device in bytes */
    int major;                       /**< Major compute capability */
    int minor;                       /**< Minor compute capability */
    int multiProcessorCount;         /**< Number of multiprocessors on device */
    int computeMode;                 /**< Compute mode (See ::cudaComputeMode) */
    int concurrentKernels;           /**< Device can possibly execute multiple kernels concurrently */
    int asyncEngineCount;            /**< Number of asynchronous engines */
    int l2CacheSize;                 /**< Size of L2 cache in bytes */
    int maxThreadsPerMultiProcessor; /**< Maximum resident threads per multiprocessor */
    int sharedMemPerMultiprocessor;  /**< Shared memory available per multiprocessor in bytes */
    int regsPerMultiprocessor;       /**< 32-bit registers available per multiprocessor */
} gpu_info_t;

typedef float MyType;
typedef float2 MyType2;
typedef float3 MyType3;
typedef float4 MyType4;

typedef cuFloatComplex MyComplex;

typedef int2 MyType2I;
typedef int3 MyType3I;
typedef int4 MyType4I;

typedef unsigned int MyUint;

#endif