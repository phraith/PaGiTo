#include "common/fitting_parameter.h"
#include <stdexcept>

FittingParameter::FittingParameter(std::string name, MyType2 mean_bounds, MyType2 stddev_bounds)
    :
    name_(name),
    mean_bounds_(mean_bounds),
    stddev_bounds_(stddev_bounds)
{
}

MyType2 FittingParameter::MeanBounds() const
{
    return mean_bounds_;
}

MyType2 FittingParameter::StddevBounds() const
{
    return stddev_bounds_;
}

const std::string& FittingParameter::Name() const
{
    return name_;
}
