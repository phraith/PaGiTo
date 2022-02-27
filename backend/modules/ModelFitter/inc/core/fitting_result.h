#ifndef MODEL_FITTER_CORE_FITTING_RESULT_H
#define MODEL_FITTER_CORE_FITTING_RESULT_H

#include <string>
#include <vector>

#include "common/standard_defs.h"

class FittingResult
{
public:
	FittingResult(std::string uuid, bool is_last, std::vector<FittedShape> fitted_shapes, SimData sim_data, std::vector<TimeMeasurement> device_timings, MyType fitting_time);

	const std::string& Uuid() const;
	bool IsLast();

	const std::vector<FittedShape>& FittedShapes() const;
	const SimData& Data() const;
	const std::vector<TimeMeasurement>& DeviceTimings() const;

	MyType FittingTime() const;

private:
	std::string uuid_;
	std::vector<FittedShape> fitted_shapes_;
	SimData sim_data_;
	bool is_last_;

	std::vector<TimeMeasurement> device_timings_;
	MyType fitting_time_;
};

#endif