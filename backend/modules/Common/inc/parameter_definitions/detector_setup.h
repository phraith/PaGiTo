//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_DETECTOR_CONFIGURATION_H
#define GISAXSMODELINGFRAMEWORK_DETECTOR_CONFIGURATION_H


#include <string>
#include "data_containers.h"

class DetectorConfiguration {
public:
    explicit DetectorConfiguration(GisaxsTransformationContainer::DetectorContainer detector_container);
    bool operator== (const DetectorConfiguration& detector_setup) const;

public:
    [[nodiscard]] MyType Pixelsize() const;

    [[nodiscard]] MyType SampleDistance() const;

    [[nodiscard]] const Vector2<int> &Directbeam() const;

    [[nodiscard]] const Vector2<int> &Resolution() const;

    [[nodiscard]] int PixelCount() const;

    [[nodiscard]] std::string InfoStr() const;
private:
    MyType sample_distance_;
    Vector2<int> directbeam_;
    Vector2<int> resolution_;
    MyType pixelsize_;
};


#endif //GISAXSMODELINGFRAMEWORK_DETECTOR_SETUP_H
