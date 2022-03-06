//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_SETUP_H
#define GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_SETUP_H


#include "detector_setup.h"
#include "scattering_setup.h"
#include "substrate_setup.h"

class ExperimentalSetup {
public:
    ExperimentalSetup(const DetectorConfiguration &detector_setup, ScatteringSetup scattering_setup, SubstrateSetup substrate_setup);

private:
    const DetectorConfiguration detector_setup_;
public:
    const DetectorConfiguration &DetectorParameters();

    const ScatteringSetup &ScatteringParameters();

    const SubstrateSetup &SubstrateParameters();

private:
    const ScatteringSetup scattering_setup_;
    const SubstrateSetup substrate_setup_;
};


#endif //GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_SETUP_H
