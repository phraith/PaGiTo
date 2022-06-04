#include <mutex>
#include "util/hardware_information.h"
#include "util/cpu_information.h"

#ifdef CUDA_GPUS_AVAILABLE

#include "gpu/core/gpu_information.h"


using namespace GpuInfo;

#endif //CUDA_GPUS_AVAILABLE


HardwareInformation::HardwareInformation()
	:
	cpu_info_(CpuInfo::GetCpuInfo())
{

#ifdef CUDA_GPUS_AVAILABLE
	for (const auto &gpu : GetGpuInfo())
	{
		devices_.emplace_back(gpu);
	}
#endif //CUDA_GPUS_AVAILABLE
    devices_.emplace_back(std::make_shared<CpuDevice>());
}

HardwareInformation::~HardwareInformation()
{
    auto f = 5;
}

Device* HardwareInformation::FindFreeDevice() const
{
	for (const auto& gpu : devices_)
	{
		if (gpu->Status() == WorkStatus::kIdle)
			return gpu.get();
	}

	return nullptr;
}

Device &HardwareInformation::LockAndReturnDevice() const {
    auto lk = std::unique_lock<std::mutex>(mutex_);

    Device *device = nullptr;

    while(device == nullptr)
    {
        cv_.wait(lk, [&] {
            device = FindFreeDevice();
            return device != nullptr;
        });
    }


    device->SetStatus(WorkStatus::kWorking);

    return *device;
}

void HardwareInformation::UnlockDevice(Device &device) const {
    auto lk = std::unique_lock<std::mutex>(mutex_);
    device.SetStatus(WorkStatus::kIdle);
    cv_.notify_one();
}

const std::vector<CpuInfo::cpu_info_t>& HardwareInformation::CpuInfo() const
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
