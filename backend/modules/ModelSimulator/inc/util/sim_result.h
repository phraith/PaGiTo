#ifndef MODEL_SIMULATOR_UTIL_SIM_RESULT_H
#define MODEL_SIMULATOR_UTIL_SIM_RESULT_H

#include <string>
#include <vector>

#include "common/standard_defs.h"

class SimResult
{
public:
	SimResult(std::string uuid, bool is_last, SimData sim_data, std::vector<TimeMeasurement> device_timings);

	const std::string& Uuid() const;
	bool IsLast();

	const SimData& Data() const;
	const std::vector<TimeMeasurement>& DeviceTimings() const;
private:
	std::string uuid_;
	SimData sim_data_;
	bool is_last_;

	std::vector<TimeMeasurement> device_timings_;
};

#endif