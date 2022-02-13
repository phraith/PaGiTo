#ifndef MODEL_SIMULATOR_UTIL_GPU_DEVICE_H
#define MODEL_SIMULATOR_UTIL_GPU_DEVICE_H

#include <cuda_runtime.h>
#include "curand.h"
#include "curand_kernel.h"
#include "gpu/core/gisaxs_functions.h"

#include "gpu/core/gpu_memory_provider.h"
#include "gpu/core/event_provider.h"
#include "gpu/core/stream_provider.h"
#include "common/device.h"
#include "common/timer.h"

#include "common/standard_defs.h"
#include "standard_vector_types.h"

typedef struct gpu_info_t
{
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

class GpuDevice : public Device
{
public:
    GpuDevice(gpu_info_t info, int device_id);
    ~GpuDevice();

    SimData RunGISAXS(const SimJob& descr, const ImageData *real_img, bool copy_intensities);

    void SetStatus(WorkStatus status) const;
    WorkStatus Status() const;
    void CleanUp();
    void ResetTimers();

    double AverageKernelTime() const;
    double AverageFullTime() const;

    double KernelTime() const;
    double FullTime() const;

    int Runs() const;

    std::string Name() const;

private:

    int Bind() const;
    
    int DeviceID() const;

    void GenerateRandoms(float* rands, int size, float mean, float stddev) const;

    std::shared_ptr<Stream> ProvideStream();

    void UnlockAllMemory();
    void UnlockAllEvents();
    void UnlockAllStreams();

    gpu_info_t info_;
    int device_id_;

    mutable WorkStatus work_status_;
    mutable int workers;

    DevUnitcell** dev_unitcell_;

    MyType fitness_;
    MyType* dev_fitness_;

    MyType scale_prod_;
    MyType* dev_scale_prod_;

    MyType scale_denom_;
    MyType* dev_scale_denom_;

    curandGenerator_t gen_;

    EventProvider event_provider_;
    StreamProvider stream_provider_;

    mutable int runs_;
    mutable double complete_runtime_;
    mutable double kernel_runtime_;

    Timer kernel_timer_;
    Timer complete_timer_;
};

#endif