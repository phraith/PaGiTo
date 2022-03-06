//
// Created by Phil on 26.02.2022.
//

#include "cpu_device.h"

#include "common/standard_defs.h"
#include "common/standard_constants.h"
#include "gisaxs_cpu_core.h"
#include "common/unitcell_utility.h"
#include "common/propagation_coefficients.h"
#include "common/qgrid.h"
#include <string>
#include <cmath>

SimData CpuDevice::RunGISAXS(const SimJob &description, const ImageData *real_img, bool copy_intensities) {
    auto flat_unitcell = description.ExperimentInfo().Unitcell();

    auto randoms = std::vector<MyType>(
            COHERENCY_DRAW_RATIO.x * COHERENCY_DRAW_RATIO.y * flat_unitcell.Parameters().size());
    for (int i = 0; i < COHERENCY_DRAW_RATIO.x * COHERENCY_DRAW_RATIO.y; ++i) {
        randoms[i] = normal_distribution_(generator_);
    }

    auto qgrid = QGrid(description.ExperimentInfo().DetectorConfig(), std::vector<int>(),
                       description.ExperimentInfo().BeamConfig(), 0);
    std::vector<std::complex<MyType>> sfs = GisaxsCpuCore::CalculateStructureFactors(qgrid.QPointsXY(),
                                                                                     qgrid.QPointsZCoeffs(),
                                                                                     flat_unitcell.Translation(),
                                                                                     flat_unitcell.Repetitions());

    auto prop_coefficients = PropagationCoefficientsCpu::PropagationCoeffsTopBuried(
            description.ExperimentInfo().SampleConfig(), description.ExperimentInfo().DetectorConfig(),
            description.ExperimentInfo().BeamConfig());
    auto intensities = GisaxsCpuCore::CalculateIntensities(qgrid.QPar(), qgrid.Q(), qgrid.QPointsXY(),
                                                           qgrid.QPointsZCoeffs(),
                                                           prop_coefficients,
                                                           flat_unitcell, randoms, sfs);

    MyType max_val = *max_element(intensities.begin(), intensities.end());

    MyType logmax = std::log10(max_val);
    //MyType logmin = std::log10(std::CalculateMaximumIntensity(2.f, 1e-10f * max_val));
    MyType logmin = 0;

    std::vector<unsigned char> normalized_intensities(intensities.size());
    for (int i = 0; i < intensities.size(); ++i) {
        MyType intensity_entry = intensities[i];
        auto log_val = std::log10(intensity_entry);
        log_val -= logmin;
        log_val /= (logmax - logmin);
        log_val = std::max(0.f, log_val);
        normalized_intensities[i] = (unsigned char) (log_val * 255.0);
    }

    return {0, std::vector<float>(), normalized_intensities, std::vector<float>(),
            std::vector<float>(), std::vector<float>(), description.ExperimentInfo().DetectorConfig().Resolution(), 0};
}

void CpuDevice::SetStatus(WorkStatus status) const {
    work_status_ = status;
}

WorkStatus CpuDevice::Status() const {
    return work_status_;
}

void CpuDevice::CleanUp() {

}

void CpuDevice::ResetTimers() {

}

double CpuDevice::AverageKernelTime() const {
    return 0;
}

double CpuDevice::AverageFullTime() const {
    return 0;
}

double CpuDevice::KernelTime() const {
    return 0;
}

double CpuDevice::FullTime() const {
    return 0;
}

int CpuDevice::Runs() const {
    return 0;
}

std::string CpuDevice::Name() const {
    return {};
}

CpuDevice::CpuDevice()
        :
        work_status_(WorkStatus::kIdle),
        normal_distribution_(0, 1) {
}
