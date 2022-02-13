#include "util/hardware_information.h"

#include <iostream>

#ifdef CUDA_GPUS_AVAILABLE

#include "gpu/core/gpu_information.h"
using namespace gpu_info;

#endif CUDA_GPUS_AVAILABLE

using namespace cpu_info;


HardwareInformation::HardwareInformation()
	:
	cpu_info_(GetCpuInfo())
{
#ifdef CUDA_GPUS_AVAILABLE

	for (const auto &gpu : GetGpuInfo())
	{
		devices_.emplace_back(gpu);
	}
#endif CUDA_GPUS_AVAILABLE
}

HardwareInformation::~HardwareInformation()
{
}

Device* HardwareInformation::FindFreeDevice()
{
	for (const auto& gpu : devices_)
	{
		if (gpu->Status() == WorkStatus::kIdle)
			return gpu.get();
	}

	return nullptr;
}

const std::vector<cpu_info_t>& HardwareInformation::CpuInfo() const
{
	return cpu_info_;
}

std::vector<std::shared_ptr<Device>>& HardwareInformation::DeviceInfo()
{
	return devices_;
}

void HardwareInformation::CleanUpDevices()
{
	for (auto& device : devices_)
	{
		device->CleanUp();
	}
}

void HardwareInformation::ResetTimers()
{
	for (auto& device : devices_)
	{
		device->ResetTimers();
	}
}