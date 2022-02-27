#ifndef MODEL_FITTER_CORE_EXPERIMENTAL_MODEL_H
#define MODEL_FITTER_CORE_EXPERIMENTAL_MODEL_H

#include <vector>
#include <memory>
#include <map>

#include <common/fitting_parameter.h>
#include <common/detector.h>
#include <common/beam_configuration.h>
#include <common/sample.h>

#include <common/image_data.h>
#include <common/qgrid.h>
#include "standard_vector_types.h"
#include "parameter_definitions/experimental_setup.h"

class ExperimentalModel {
public:
    ExperimentalModel(DetectorSetup detector, std::vector<int> position_offsets, BeamConfiguration beam_config,
                      Sample sample);

    [[nodiscard]] const std::vector<MyComplex> &GetPropagationCoefficients() const;

    DetectorSetup detector_;
    const std::vector<int> position_offsets_;
    BeamConfiguration beam_config_;
    Sample sample_;
    double sample_detector_dist_;
private:
    const std::vector<MyComplex> prop_coeffs_;

};

#endif