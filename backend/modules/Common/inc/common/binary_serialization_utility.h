//
// Created by Phil on 27.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_BINARY_SERIALIZATION_UTILITY_H
#define GISAXSMODELINGFRAMEWORK_BINARY_SERIALIZATION_UTILITY_H

#include <cstddef>
#include <vector>
#include "parameter_definitions/data_containers.h"

namespace BinarySerializationUtility {
    std::vector<GisaxsTransformationContainer::LineProfileContainer>
    ReadLineProfiles(const std::vector<std::byte> &data);

    std::vector<GisaxsTransformationContainer::SimulationTargetData>
    ReadSimulationTargetData(const std::vector<std::byte> &data);
}


#endif //GISAXSMODELINGFRAMEWORK_BINARY_SERIALIZATION_UTILITY_H
