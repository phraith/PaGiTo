//
// Created by Phil on 09.01.2022.
//

#include "parameter_definitions/detector_setup.h"

DetectorSetup::DetectorSetup(MyType pixelsize, MyType sampleDistance, MyType2I beamImpact, MyType2I resolution)
        : pixelsize_(pixelsize), sample_distance_(sampleDistance), directbeam_(beamImpact), resolution_(resolution) {

}

bool DetectorSetup::operator==(const DetectorSetup &detector_setup) {
    return
            pixelsize_ == detector_setup.pixelsize_ &&
            sample_distance_ == detector_setup.sample_distance_ &&
            directbeam_.x == detector_setup.directbeam_.x &&
            directbeam_.y == detector_setup.directbeam_.y &&
            resolution_.x == detector_setup.resolution_.x &&
            resolution_.y == detector_setup.resolution_.y;
}

MyType DetectorSetup::Pixelsize() const {
    return pixelsize_;
}

MyType DetectorSetup::SampleDistance() const {
    return sample_distance_;
}

const MyType2I &DetectorSetup::Directbeam() const {
    return directbeam_;
}

const MyType2I &DetectorSetup::Resolution() const {
    return resolution_;
}

std::string DetectorSetup::InfoStr() const
{
    std::string info = "";
    info += "Detector info:\n";
    info += "	-pixel_size in mm: " + std::to_string(pixelsize_) + "\n";
    info += "	-resolution (x, y): " + std::to_string(resolution_.x) + ", " + std::to_string(resolution_.y) + "\n";
    info += "	-direct beam location in mm (x, y): " + std::to_string(directbeam_.x * pixelsize_) + ", " + std::to_string(directbeam_.y * pixelsize_) + "\n";
    return info;
}

DetectorSetup::DetectorSetup() {

}
