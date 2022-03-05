//
// Created by Phil on 26.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_CPU_DEVICE_H
#define GISAXSMODELINGFRAMEWORK_CPU_DEVICE_H


#include <random>
#include "common/device.h"

class CpuDevice : public Device {
public:
    CpuDevice();
    SimData RunGISAXS(const SimJob &description, const ImageData *real_img, bool copy_intensities) override;
    void SetStatus(WorkStatus status) const override;
    [[nodiscard]] WorkStatus Status() const override;

    void CleanUp() override;
    void ResetTimers() override;

    [[nodiscard]] double AverageKernelTime() const override;
    [[nodiscard]] double AverageFullTime() const override;

    [[nodiscard]] double KernelTime() const override;
    [[nodiscard]] double FullTime() const override;

    [[nodiscard]] int Runs() const override;

    [[nodiscard]] std::string Name() const override;
private:
    std::normal_distribution<MyType> normal_distribution_;
    std::default_random_engine generator_;
    mutable WorkStatus work_status_;
};


#endif //GISAXSMODELINGFRAMEWORK_CPU_DEVICE_H
