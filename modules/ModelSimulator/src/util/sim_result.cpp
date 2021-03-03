#include "util/sim_result.h"

SimResult::SimResult(std::string uuid, bool is_last, SimData sim_data, std::vector<TimeMeasurement> device_timings)
	:
	uuid_(uuid),
	sim_data_(sim_data),
	is_last_(is_last),
	device_timings_(device_timings)
{
}

const std::string& SimResult::Uuid() const
{
	return uuid_;
}

bool SimResult::IsLast()
{
	return is_last_;
}

const SimData& SimResult::Data() const
{
	return sim_data_;
}

const std::vector<TimeMeasurement>& SimResult::DeviceTimings() const
{
	return device_timings_;
}
