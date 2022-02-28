//
// Created by Phil on 09.01.2022.
//

#include "parameter_definitions/experimental_setup.h"

ExperimentalSetup::ExperimentalSetup(const DetectorConfiguration &detector_setup, ScatteringSetup scattering_setup,
                                     SubstrateSetup substrate_setup)
        :
        detector_setup_(detector_setup),
        scattering_setup_(scattering_setup),
        substrate_setup_(substrate_setup)
        {}

const DetectorConfiguration &ExperimentalSetup::DetectorParameters()  {
    return detector_setup_;
}

const ScatteringSetup &ExperimentalSetup::ScatteringParameters() {
    return scattering_setup_;
}

const SubstrateSetup &ExperimentalSetup::SubstrateParameters() {
    return substrate_setup_;
}
