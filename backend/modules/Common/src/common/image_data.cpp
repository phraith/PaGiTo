#include "common/image_data.h"
#include "parameter_definitions/data_containers.h"

#include <iostream>
#include <algorithm>


ImageData::ImageData(std::vector<GisaxsTransformationContainer::SimulationTargetData> simulation_targets)
        :
        simulation_targets_(std::move(simulation_targets)) {
}

ImageData::~ImageData() = default;

const std::vector<GisaxsTransformationContainer::SimulationTargetData> &ImageData::SimulationTargets() const {
    return simulation_targets_;
}

std::vector<MyType> ImageData::CombinedSimulationTargetIntensities() const {
    size_t combined_size = 0;
    for (const auto &target: simulation_targets_) {
        combined_size += target.intensities.size();
    }

    std::vector<MyType> combined_intensities;
    combined_intensities.reserve(combined_size);

    for (const auto &target: simulation_targets_) {
        combined_intensities.insert(combined_intensities.end(), target.intensities.begin(), target.intensities.end());
    }

    return combined_intensities;
}
