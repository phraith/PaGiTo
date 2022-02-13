#include "common/experimental_model.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>


ExperimentalModel::ExperimentalModel(DetectorSetup detector, const std::vector<int> position_offsets,
                                     BeamConfiguration beam_config, Sample sample, double sample_detector_dist,
                                     int level)
        :
        detector_(detector),
        position_offsets_(position_offsets),
        beam_config_(beam_config),
        sample_(sample),
        sample_detector_dist_(sample_detector_dist),
        level_(level),
        //prop_coeffs_(sample_.PropagationCoeffs(beam_config_.AlphaI(), qgrid_.AlphaFs(), beam_config_.K0(), level))
        prop_coeffs_(sample_.PropagationCoeffsTopBuried(detector_, beam_config_)) {
}

const std::vector<MyComplex> &ExperimentalModel::GetPropagationCoefficients() const {
    return prop_coeffs_;
}


