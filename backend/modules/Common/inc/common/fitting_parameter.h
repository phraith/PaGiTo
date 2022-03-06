#ifndef MODEL_SIMULATOR_CORE_FITTING_PARAMETER_H
#define MODEL_SIMULATOR_CORE_FITTING_PARAMETER_H

#include <string>

#include "standard_defs.h"

class FittingParameter
{
public:
	FittingParameter(std::string name, Vector2<MyType> mean_bounds, Vector2<MyType> stddev_bounds);

    [[nodiscard]] Vector2<MyType> MeanBounds() const;
    [[nodiscard]] Vector2<MyType> StddevBounds() const;

	[[nodiscard]] const std::string& Name() const;
private:
	std::string name_;
    Vector2<MyType> mean_bounds_;
    Vector2<MyType> stddev_bounds_;
};

#endif