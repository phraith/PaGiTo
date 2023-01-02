#ifndef MODEL_SIMULATOR_UTIL_IMAGE_DATA_H
#define MODEL_SIMULATOR_UTIL_IMAGE_DATA_H

#include <vector>
#include <string>

#include "parameter_definitions/transformation_container.h"

class ImageData {
public:
    explicit ImageData(std::vector<GisaxsTransformationContainer::SimulationTargetData> simulation_targets);

    ~ImageData();

    const std::vector<GisaxsTransformationContainer::SimulationTargetData> &SimulationTargets() const;

    std::vector<MyType> CombinedSimulationTargetIntensities() const;

private:
    std::vector<GisaxsTransformationContainer::SimulationTargetData> simulation_targets_;
};

#endif