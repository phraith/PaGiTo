#ifndef MODEL_SIMULATOR_UTIL_GPU_DEVICE_H
#define MODEL_SIMULATOR_UTIL_GPU_DEVICE_H

#include "gpu/core/gisaxs_functions.h"
#include "gpu/core/gpu_qgrid.h"

#include "gpu/core/gpu_memory_provider.h"
#include "gpu/core/event_provider.h"
#include "gpu/core/stream_provider.h"
#include "common/device.h"
#include "common/timer.h"

#include "common/standard_defs.h"
#include "random_generator.h"

class GpuDevice : public Device
{
public:
    GpuDevice(gpu_info_t info, int device_id);
    ~GpuDevice();

    SimData RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities);

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

    EventProvider event_provider_;
    StreamProvider stream_provider_;
    RandomGenerator random_generator_;

    mutable int runs_;
    mutable double complete_runtime_;
    mutable double kernel_runtime_;

    Timer kernel_timer_;
    Timer complete_timer_;
};

#endif