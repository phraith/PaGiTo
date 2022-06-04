#include "common/image_data.h"
#include "parameter_definitions/data_containers.h"

#include <iostream>
#include <algorithm>

using namespace GisaxsTransformationContainer;

ImageData::ImageData(std::vector<LineProfileContainer> line_profiles)
        :
        line_profiles_(std::move(line_profiles)) {
}

ImageData::~ImageData() = default;

const std::vector<LineProfileContainer> &ImageData::LineProfiles() const {
    return line_profiles_;
}
