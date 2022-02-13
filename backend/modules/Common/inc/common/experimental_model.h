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
    //ExperimentalModel(ExperimentalSetup setup);

    ExperimentalModel(DetectorSetup detector, std::vector<int> position_offsets, BeamConfiguration beam_config,
                      Sample sample, double sample_detector_dist, int level);

    const QGrid &GetQGrid() const;

    const std::vector<MyComplex> &GetPropagationCoefficients() const;

    void PrintInfo() const;

private:

    std::string DwbaInfo(int idx) const;


    //ExperimentalSetup experimental_setup_;
    DetectorSetup detector_;
    const std::vector<int> position_offsets_;
    BeamConfiguration beam_config_;
    Sample sample_;
    double sample_detector_dist_;

    QGrid qgrid_;
    const std::vector<MyComplex> prop_coeffs_;
    int level_;
};

#endif