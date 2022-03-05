//
// Created by Phil on 09.01.2022.
//

#include "parameter_definitions/detector_setup.h"
#include "parameter_definitions/transformation_container.h"

using namespace GisaxsTransformationContainer;

DetectorConfiguration::DetectorConfiguration(DetectorContainer detector_container)
        : pixelsize_(detector_container.pixelsize), sample_distance_(detector_container.sampleDistance),
          directbeam_(detector_container.beamImpact), resolution_(detector_container.resolution) {

}

bool DetectorConfiguration::operator==(const DetectorConfiguration &detector_setup) const {
    return
            pixelsize_ == detector_setup.pixelsize_ &&
            sample_distance_ == detector_setup.sample_distance_ &&
            directbeam_.x == detector_setup.directbeam_.x &&
            directbeam_.y == detector_setup.directbeam_.y &&
            resolution_.x == detector_setup.resolution_.x &&
            resolution_.y == detector_setup.resolution_.y;
}

MyType DetectorConfiguration::Pixelsize() const {
    return pixelsize_;
}

MyType DetectorConfiguration::SampleDistance() const {
    return sample_distance_;
}

const MyType2I &DetectorConfiguration::Directbeam() const {
    return directbeam_;
}

const MyType2I &DetectorConfiguration::Resolution() const {
    return resolution_;
}

std::string DetectorConfiguration::InfoStr() const {
    std::string info;
    info += "DetectorConfiguration info:\n";
    info += "	-pixel_size in mm: " + std::to_string(pixelsize_) + "\n";
    info += "	-resolution (x, y): " + std::to_string(resolution_.x) + ", " + std::to_string(resolution_.y) + "\n";
    info += "	-direct beam location in mm (x, y): " + std::to_string(directbeam_.x * pixelsize_) + ", " +
            std::to_string(directbeam_.y * pixelsize_) + "\n";
    return info;
}

int DetectorConfiguration::PixelCount() const {
    return resolution_.x * resolution_.y;
}
