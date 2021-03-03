#include "util/distribution_functions.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "common/fitting_parameter.h"
#include "core/simulation_interval.h"

namespace distribution_functions
{
	const DistFuncMap dist_func_map = 
	{
		{DistributionType::kEVEN, DistributeEven}
	};

	const std::vector<SimulationInterval> DistributeEven(std::shared_ptr<SimJob> simDescr, GpuDevice &device, int blocksize)
	{
		if (blocksize <= 0)
			throw std::invalid_argument("Blocksize is smaller than or equal zero!");

		std::vector<SimulationInterval> intervals;

		//if (simDescr->Size() == 0)
		//	return std::move(intervals);

		//for (int i = 0; i <= std::ceil(simDescr->Size() / blocksize); ++i)
		//{
		//	intervals.emplace_back(SimulationInterval( std::vector<FittingParameter>{}, device, i * blocksize, std::min((i + 1) * blocksize - 1, simDescr->Size() - 1) ));
		//}
		return std::move(intervals);
	}
}