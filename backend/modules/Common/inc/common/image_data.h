#ifndef MODEL_SIMULATOR_UTIL_IMAGE_DATA_H
#define MODEL_SIMULATOR_UTIL_IMAGE_DATA_H

#include <vector>
#include <string>

#include "vector_types.h"

#include "uuid_generator.h"

class ImageData
{
public:
    ImageData(std::vector<float> intensities, std::vector<int> offsets);
    ~ImageData();

    const std::vector<float> &Intensities() const;
    const std::vector<int>& Offsets() const;
    const std::string& Id() const ;

    int Size() const;
    float MaxIntensity() const;
private:
    std::vector<float> intensities_;
    float max_intensity_;
    float min_intensity_;
    std::vector<int> offsets_;
    std::string uuid_;
};

#endif