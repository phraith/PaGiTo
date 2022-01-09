#ifndef MODEL_SIMULATOR_CORE_FITTING_PARAMETER_H
#define MODEL_SIMULATOR_CORE_FITTING_PARAMETER_H

#include <string>

#include "standard_vector_types.h"

class FittingParameter
{
public:
	FittingParameter(std::string name, MyType2 mean_bounds, MyType2 stddev_bounds);

	MyType2 MeanBounds() const;
	MyType2 StddevBounds() const;

	const std::string& Name() const;
private:
	std::string name_;
	MyType2 mean_bounds_;
	MyType2 stddev_bounds_;
};

#endif