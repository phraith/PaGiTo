//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_DETECTOR_SETUP_H
#define GISAXSMODELINGFRAMEWORK_DETECTOR_SETUP_H


#include <string>
#include "standard_vector_types.h"

class DetectorSetup {
public:
    DetectorSetup();
    DetectorSetup(MyType pixelsize, MyType sampleDistance, MyType2I beamImpact, MyType2I resolution);
    bool operator== (const DetectorSetup& detector_setup);

public:
    MyType Pixelsize() const;

    MyType SampleDistance() const;

    const MyType2I &Directbeam() const;

    const MyType2I &Resolution() const;

    std::string InfoStr() const;
private:
    MyType sample_distance_{};
    MyType2I directbeam_{};
    MyType2I resolution_{};
    MyType pixelsize_{};
};


#endif //GISAXSMODELINGFRAMEWORK_DETECTOR_SETUP_H
