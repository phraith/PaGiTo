#ifndef MODEL_SIMULATOR_UTIL_IMAGE_DATA_H
#define MODEL_SIMULATOR_UTIL_IMAGE_DATA_H

#include <vector>
#include <string>

#include "parameter_definitions/transformation_container.h"

class ImageData {
public:
    explicit ImageData(std::vector<GisaxsTransformationContainer::LineProfileContainer> line_profiles);

    ~ImageData();

    const std::vector<GisaxsTransformationContainer::LineProfileContainer> &LineProfiles() const;

private:
    std::vector<GisaxsTransformationContainer::LineProfileContainer> line_profiles_;
};

#endif