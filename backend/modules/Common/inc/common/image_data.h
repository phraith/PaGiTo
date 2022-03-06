#ifndef MODEL_SIMULATOR_UTIL_IMAGE_DATA_H
#define MODEL_SIMULATOR_UTIL_IMAGE_DATA_H

#include <vector>
#include <string>

#include "uuid_generator.h"

class ImageData
{
public:
    ImageData(std::vector<float> intensities, std::vector<int> offsets);
    ~ImageData();

    [[nodiscard]] const std::vector<float> &Intensities() const;
    [[nodiscard]] const std::vector<int>& Offsets() const;
    [[nodiscard]] const std::string& Id() const ;

    [[nodiscard]] int Size() const;
    [[nodiscard]] float MaxIntensity() const;
private:
    std::vector<float> intensities_;
    float max_intensity_;
    float min_intensity_;
    std::vector<int> offsets_;
    std::string uuid_;
};

#endif