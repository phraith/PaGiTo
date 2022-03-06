#include "common/fitting_parameter.h"
#include <stdexcept>
#include <utility>

FittingParameter::FittingParameter(std::string name, Vector2<MyType> mean_bounds, Vector2<MyType> stddev_bounds)
    :
    name_(std::move(name)),
    mean_bounds_(mean_bounds),
    stddev_bounds_(stddev_bounds)
{
}

Vector2<MyType> FittingParameter::MeanBounds() const
{
    return mean_bounds_;
}

Vector2<MyType> FittingParameter::StddevBounds() const
{
    return stddev_bounds_;
}

const std::string& FittingParameter::Name() const
{
    return name_;
}
