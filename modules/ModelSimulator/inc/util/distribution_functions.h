#ifndef MODEL_SIMULATOR_UTIL_DISTRIBUTION_FUNCTIONS_H
#define MODEL_SIMULATOR_UTIL_DISTRIBUTION_FUNCTIONS_H

#include <vector>
#include <map>

#include "common/simulation_description.h"
#include "core/simulation_interval.h"

namespace distribution_functions
{
	enum class DistributionType {kEVEN};

	typedef const std::vector<SimulationInterval> (*DistFunc)(std::shared_ptr<SimJob> simDescr, GpuDevice& devices, int blocksize);
	typedef std::map<DistributionType, DistFunc> DistFuncMap;

	extern const DistFuncMap dist_func_map;

	const std::vector<SimulationInterval> DistributeEven(std::shared_ptr<SimJob> simDescr, GpuDevice& device, int blocksize);
}

#endif
