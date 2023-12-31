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
#include <iostream>
#include <spdlog/spdlog.h>

SimData CpuDevice::RunGISAXS(const SimJob &description, std::shared_ptr<ImageData> real_img, bool copy_intensities) {
    auto flat_unitcell = description.ExperimentInfo().Unitcell();

    auto randoms = std::vector<MyType>(
            COHERENCY_DRAW_RATIO.x * COHERENCY_DRAW_RATIO.y * flat_unitcell.Parameters().size());
    for (int i = 0; i < randoms.size(); ++i) {
        randoms[i] = normal_distribution_(generator_);
    }

    auto eConfig = description.ExperimentInfo();
    auto dConfig = eConfig.DetectorConfig();
    auto bConfig = eConfig.BeamConfig();

    bool isFull = description.JobInfo().SimulationTargets().size() == 0;

    auto qgrid = isFull ? QGrid(dConfig, std::vector<Vector2<int>>(), bConfig, 0) : QGrid(dConfig,
                                                                                          description.DetectorPositions(),
                                                                                          bConfig, 0);

    std::vector<std::complex<MyType>> sfs = GisaxsCpuCore::CalculateStructureFactors(qgrid.QPointsXY(),
                                                                                     qgrid.QPointsZCoeffs(),
                                                                                     flat_unitcell.Translation(),
                                                                                     flat_unitcell.Repetitions());

    auto prop_coefficients = isFull ? PropagationCoefficientsCpu::PropagationCoeffsTopBuriedFull(
            description.ExperimentInfo().SampleConfig(), description.ExperimentInfo().DetectorConfig(),
            description.ExperimentInfo().BeamConfig()) : PropagationCoefficientsCpu::PropagationCoeffsTopBuried(
            description.ExperimentInfo().SampleConfig(), description.DetectorPositions(),
            description.ExperimentInfo().DetectorConfig(),
            description.ExperimentInfo().BeamConfig());

    auto intensities = GisaxsCpuCore::CalculateIntensities(qgrid.QPar(), qgrid.Q(), qgrid.QPointsXY(),
                                                           qgrid.QPointsZCoeffs(),
                                                           prop_coefficients,
                                                           flat_unitcell, randoms, sfs);

    MyType max_val = *max_element(intensities.begin(), intensities.end());
    spdlog::info(max_val);
    MyType logmax = std::log(max_val);
    MyType logmin = std::log(std::max(2.f, 1e-10f * max_val));

    std::vector<unsigned char> normalized_intensities(intensities.size());
    for (int i = 0; i < intensities.size(); ++i) {
        MyType intensity_entry = intensities[i];
        auto log_val = std::log(intensity_entry);
        log_val -= logmin;
        log_val /= (logmax - logmin);
        log_val = std::max(0.f, log_val);
        normalized_intensities[i] = (unsigned char) (log_val * 255.0);
    }


    float scale = 1;
    if (real_img != nullptr) {

        std::vector<MyType> combined_intensities = real_img->CombinedSimulationTargetIntensities();

        float scale_prod = 0;
        std::for_each(combined_intensities.begin(), combined_intensities.end(), [&](int n) {
            scale_prod += n;
        });

        float scale_denom = 0;
        std::for_each(intensities.begin(), intensities.end(), [&](int n) {
            scale_denom += n;
        });

        //scale = scale_prod / scale_denom;

        float fitness =  0;
        int j = 0;
        std::for_each(intensities.begin(), intensities.end(), [&](int n) {
            float real_value = combined_intensities.at(j);

            float abs_diff = real_value - n * scale;
            fitness += (abs_diff * abs_diff);
            j++;
        });
        spdlog::info("Fitness: {}", fitness);
        return {fitness, {}, std::vector<unsigned char>(), {}, {}, {}, 0};
    }


    return {0, std::vector<double>(intensities.begin(), intensities.end()), normalized_intensities,
            std::vector<float>(),
            std::vector<float>(), std::vector<float>(), 0};
}

void CpuDevice::SetStatus(WorkStatus status) const {
    spdlog::info("Cpu: status {}", WorkStatusToStr(status));
    work_status_ = status;
}

std::string CpuDevice::WorkStatusToStr(WorkStatus status) const {
    switch (status) {
        case WorkStatus::kIdle:
            return "idle";
        case WorkStatus::kWorking:
            return "working";
    }
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
