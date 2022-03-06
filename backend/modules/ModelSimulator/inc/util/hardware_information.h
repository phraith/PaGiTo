#ifndef MODEL_SIMULATOR_UTIL_CPU_INFORMATION_H
#define MODEL_SIMULATOR_UTIL_CPU_INFORMATION_H

#include <vector>

#include "util/cpu_information.h"
#include "common/device.h"
#include "cpu_device.h"

enum class DeviceType {kGPU, kCPU};

class HardwareInformation
{
public:
	HardwareInformation();
	~HardwareInformation();

	Device* FindFreeDevice();

	[[nodiscard]] const std::vector<CpuInfo::cpu_info_t>& CpuInfo() const;
	std::vector<std::shared_ptr<Device>>& DeviceInfo();

	void CleanUpDevices();
	void ResetTimers();

private:
	std::vector<CpuInfo::cpu_info_t> cpu_info_;
	std::vector<std::shared_ptr<Device>> devices_;
};

#endif