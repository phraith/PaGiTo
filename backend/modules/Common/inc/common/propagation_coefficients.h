//
// Created by Phil on 28.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H
#define GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H

#include <vector>
#include "parameter_definitions/detector_setup.h"
#include "beam_configuration.h"
#include "sample.h"

namespace PropagationCoefficientsCpu {
    std::vector<std::complex<MyType>>
    PropagationCoeffsTopBuriedFull(const SampleConfiguration &sample_config, const DetectorConfiguration &detector,
                                   const BeamConfiguration &beam_config);


    std::vector<std::complex<MyType>>
    PropagationCoeffsTopBuried(const SampleConfiguration &sample_config,
                               const std::vector<Vector2<int>> &detector_positions,
                               const DetectorConfiguration &detector,
                               const BeamConfiguration &beam_config);
}


#endif //GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H
