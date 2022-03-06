//#include "core/simulation_interval.h"
//
//SimulationInterval::SimulationInterval(const std::vector<FittingParameter> &parameters, GpuDevice& device, int first_idx, int last_idx)
//    :
//    parameters_(parameters),
//    device_(device),
//    first_idx_(first_idx),
//    last_idx_(last_idx)
//{}
//
//SimulationInterval::~SimulationInterval()
//{}
//
//
//const std::vector<FittingParameter> &SimulationInterval::Params() const
//{
//    return parameters_;
//}
//
//GpuDevice& SimulationInterval::Device() const
//{
//    return device_;
//}
//
//
//int SimulationInterval::FirstIdx() const
//{
//    return first_idx_;
//}
//
//int SimulationInterval::LastIdx() const
//{
//    return last_idx_;
//}
//
//int SimulationInterval::Size() const
//{
//    return LastIdx() - FirstIdx() + 1;
//}
