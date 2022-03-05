#ifndef MODEL_SIMULATOR_UTIL_GPU_INFORMATION_H
#define MODEL_SIMULATOR_UTIL_GPU_INFORMATION_H

#include <vector>

#include "gpu/core/gpu_device.h"
#include "gpu_device_v2.h"

namespace gpu_info
{
    std::vector<std::shared_ptr<Device>> GetGpuInfo();
}
#endif