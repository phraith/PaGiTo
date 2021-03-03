#include "core/fitting_result.h"

FittingResult::FittingResult(std::string uuid, bool is_last, std::vector<FittedShape> fitted_shapes, SimData sim_data, std::vector<TimeMeasurement> device_timings, MyType fitting_time)
	:
	uuid_(uuid),
	fitted_shapes_(fitted_shapes),
	sim_data_(sim_data),
	is_last_(is_last),
	device_timings_(device_timings),
	fitting_time_(fitting_time)
{
}

const std::string& FittingResult::Uuid() const
{
	return uuid_;
}

bool FittingResult::IsLast()
{
	return is_last_;
}

const std::vector<FittedShape>& FittingResult::FittedShapes() const
{
	return fitted_shapes_;
}

const SimData& FittingResult::Data() const
{
	return sim_data_;
}

const std::vector<TimeMeasurement>& FittingResult::DeviceTimings() const
{
	return device_timings_;
}

MyType FittingResult::FittingTime() const
{
	return fitting_time_;
}
