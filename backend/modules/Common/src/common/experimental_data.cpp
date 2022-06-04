//
// Created by Phil on 28.02.2022.
//

#include "common/experimental_data.h"
#include <utility>

ExperimentalData::ExperimentalData(const DetectorConfiguration &detector, const BeamConfiguration &beam_configuration,
                                   const SampleConfiguration &sample, FlatUnitcellV2 flat_unitcell)
                                   :
                                   detector_(detector),
                                   beam_configuration_(beam_configuration),
                                   sample_(sample),
                                   flat_unitcell_(std::move(flat_unitcell))
                                   {}

const DetectorConfiguration &ExperimentalData::DetectorConfig() const {
    return detector_;
}

const BeamConfiguration &ExperimentalData::BeamConfig() const {
    return beam_configuration_;
}

const SampleConfiguration &ExperimentalData::SampleConfig() const {
    return sample_;
}

const FlatUnitcellV2 &ExperimentalData::Unitcell() const {
    return flat_unitcell_;
}
