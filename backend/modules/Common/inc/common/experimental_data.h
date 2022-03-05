//
// Created by Phil on 28.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_DATA_H
#define GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_DATA_H


#include "parameter_definitions/detector_setup.h"
#include "common/beam_configuration.h"
#include "common/sample.h"
#include "common/flat_unitcell.h"

class ExperimentalData {
public:
    ExperimentalData(const DetectorConfiguration &detector, const BeamConfiguration &beam_configuration,
                     const SampleConfiguration &sample, FlatUnitcellV2 flat_unitcell);

    [[nodiscard]] const DetectorConfiguration &DetectorConfig() const;
    [[nodiscard]] const BeamConfiguration &BeamConfig() const;
    [[nodiscard]] const SampleConfiguration &SampleConfig() const;
    [[nodiscard]] const FlatUnitcellV2 &Unitcell() const;


private:
    DetectorConfiguration detector_;
    BeamConfiguration beam_configuration_;
    SampleConfiguration sample_;
    FlatUnitcellV2 flat_unitcell_;
};


#endif //GISAXSMODELINGFRAMEWORK_EXPERIMENTAL_DATA_H
