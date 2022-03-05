//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_DETECTOR_CONFIGURATION_H
#define GISAXSMODELINGFRAMEWORK_DETECTOR_CONFIGURATION_H


#include <string>
#include "standard_vector_types.h"
#include "transformation_container.h"

class DetectorConfiguration {
public:
    explicit DetectorConfiguration(GisaxsTransformationContainer::DetectorContainer detector_container);
    bool operator== (const DetectorConfiguration& detector_setup) const;

public:
    [[nodiscard]] MyType Pixelsize() const;

    [[nodiscard]] MyType SampleDistance() const;

    [[nodiscard]] const MyType2I &Directbeam() const;

    [[nodiscard]] const MyType2I &Resolution() const;

    [[nodiscard]] int PixelCount() const;

    [[nodiscard]] std::string InfoStr() const;
private:
    MyType sample_distance_;
    MyType2I directbeam_;
    MyType2I resolution_;
    MyType pixelsize_;
};


#endif //GISAXSMODELINGFRAMEWORK_DETECTOR_SETUP_H
