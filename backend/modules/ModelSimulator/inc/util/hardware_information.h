#ifndef MODEL_SIMULATOR_UTIL_CPU_INFORMATION_H
#define MODEL_SIMULATOR_UTIL_CPU_INFORMATION_H

#include <vector>
#include <thread>
#include <condition_variable>
#include "util/cpu_information.h"
#include "common/device.h"
#include "cpu_device.h"

enum class DeviceType {kGPU, kCPU};

class HardwareInformation
{
public:
	HardwareInformation();
	~HardwareInformation();

	Device* FindFreeDevice() const;
    void UnlockDevice(Device &device) const;
    Device &LockAndReturnDevice() const;

    [[nodiscard]] const std::vector<CpuInfo::cpu_info_t>& CpuInfo() const;
	std::vector<std::shared_ptr<Device>>& DeviceInfo();

	void CleanUpDevices();
	void ResetTimers();

private:
	std::vector<CpuInfo::cpu_info_t> cpu_info_;
	std::vector<std::shared_ptr<Device>> devices_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
};

#endif