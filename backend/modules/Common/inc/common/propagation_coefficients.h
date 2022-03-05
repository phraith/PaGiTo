//
// Created by Phil on 28.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H
#define GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H

#include <vector>
#include "standard_vector_types.h"
#include "parameter_definitions/detector_setup.h"
#include "beam_configuration.h"
#include "sample.h"

namespace GisaxsPropagationCoefficients {
    std::vector<MyComplex>
    PropagationCoeffsTopBuried(const SampleConfiguration &sample_config, const DetectorConfiguration &detector,
                               const BeamConfiguration &beam_config);
}


#endif //GISAXSMODELINGFRAMEWORK_PROPAGATION_COEFFICIENTS_H
