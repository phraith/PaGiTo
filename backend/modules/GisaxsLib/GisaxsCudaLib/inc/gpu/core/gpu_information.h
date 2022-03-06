#ifndef MODEL_SIMULATOR_UTIL_GPU_INFORMATION_H
#define MODEL_SIMULATOR_UTIL_GPU_INFORMATION_H

#include <vector>
#include <memory>
#include "common/device.h"

namespace gpu_info
{
    std::vector<std::shared_ptr<Device>> GetGpuInfo();
}
#endif