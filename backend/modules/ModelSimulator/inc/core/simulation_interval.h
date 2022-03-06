//#ifndef MODEL_SIMULATOR_UTIL_SIMULATION_INTERVAL_H
//#define MODEL_SIMULATOR_UTIL_SIMULATION_INTERVAL_H
//
//#include <vector>
//
//#include "common/fitting_parameter.h"
//#include "gpu/core/gpu_device.h"
//
//class SimulationInterval
//{
//public:
//    SimulationInterval(const std::vector<FittingParameter> &parameters, GpuDevice &device, int first_idx, int last_idx);
//    ~SimulationInterval();
//
//    const std::vector<FittingParameter> &Params() const;
//
//    GpuDevice& Device() const;
//
//    int FirstIdx() const;
//    int LastIdx() const;
////
//    int Size() const;
//private:
//    const std::vector<FittingParameter> &parameters_;
////
//    GpuDevice& device_;
//    int first_idx_;
//    int last_idx_;
//};
//
//#endif