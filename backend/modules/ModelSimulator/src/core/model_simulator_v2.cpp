#include "core/model_simulator_v2.h"
#include "parameter_definitions/transformation_container.h"
#include "common/unitcell_utility.h"
#include "common/timer.h"
#include "spdlog/spdlog.h"
#include <thread>
#include <algorithm>

using json = nlohmann::json;

ModelSimulatorV2::ModelSimulatorV2(std::shared_ptr<HardwareInformation> hw_info)
        :
        hw_info_(hw_info) {
}

ModelSimulatorV2::~ModelSimulatorV2() {
    hw_info_->CleanUpDevices();
}


std::vector<std::byte> ModelSimulatorV2::HandleRequest(const std::string &request, std::vector<std::byte> image_data, const std::string &origin) {
    Timer localTimer;
    Timer globalTimer;
    //spdlog::info("Received request");

    globalTimer.Start();
    localTimer.Start();

    SimJob job = GisaxsTransformationContainer::CreateSimJobFromRequest(request);
    localTimer.End();
    //spdlog::info("Preparing data took {} ms", localTimer.Duration());

    localTimer.Start();
    SimData sim_data = RunGISAXS(job, nullptr, true);
    localTimer.End();
    //spdlog::info("Simulation took {} ms", localTimer.Duration());

    localTimer.Start();

    auto final_message = CreateSerializedByteResult(sim_data, job.ExperimentInfo().DetectorConfig(), job.JobInfo());
    localTimer.End();
    //spdlog::info("Serializing result took {} ms", localTimer.Duration());

    //hw_info_->CleanUpDevices();

    globalTimer.End();
    spdlog::info("Finished request in {} ms", globalTimer.Duration());
    return final_message;
}

SimData ModelSimulatorV2::RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities) const {
    Device &device = hw_info_->LockAndReturnDevice();

    SimData sim_data = device.RunGISAXS(descr, real_img, copy_intensities);

    hw_info_->UnlockDevice(device);


    return sim_data;
}

//Device &ModelSimulatorV2::LockAndReturnDevice() const {
//    auto lk = std::unique_lock<std::mutex>(mutex_);
//
//    Device *device = nullptr;
//
//    cv_.wait(lk, [&] {
//        device = hw_info_->FindFreeDevice();
//        return device != nullptr;
//    });
//
//    device->SetStatus(WorkStatus::kWorking);
//
//    return *device;
//}
//
//void ModelSimulatorV2::UnlockDevice(Device &device) const {
//    auto lk = std::unique_lock<std::mutex>(mutex_);
//    device.SetStatus(WorkStatus::kIdle);
//}

std::vector<TimeMeasurement> ModelSimulatorV2::GetDeviceTimings() const {
    std::vector<TimeMeasurement> timings;
    for (const auto &device: hw_info_->DeviceInfo()) {
        if (device->KernelTime() != 0 && device->FullTime() != 0)
            timings.emplace_back(TimeMeasurement{device->Name(), device->KernelTime() / 1000.f, device->FullTime(),
                                                 device->AverageKernelTime() / 1000.f, device->AverageFullTime(),
                                                 device->Runs()});
    }
    return timings;
}

std::string ModelSimulatorV2::ServiceName() const {
    return "sim";
}

std::string
ModelSimulatorV2::CreateSerializedResult(const SimData &sim_data, const DetectorConfiguration &detector_data,
                                         const JobMetaInformation &job_meta_information) {

    nlohmann::json message;
    message["intensities"] = sim_data.normalized_intensities;
    message["width"] = detector_data.Resolution().x;
    message["height"] = detector_data.Resolution().y;
    message["id"] = job_meta_information.ID();

    return message.dump();
}

std::vector<std::byte>
ModelSimulatorV2::CreateSerializedByteResult(const SimData &sim_data, const DetectorConfiguration &detector_data,
                                             const JobMetaInformation &job_meta_information) {
    size_t s1 = sizeof detector_data.Resolution().x;
    size_t s2 = job_meta_information.ID().size();
    size_t s3 = sim_data.normalized_intensities.size() * sizeof sim_data.normalized_intensities[0];
    size_t s4 = sim_data.intensities.size() * sizeof(double);

    std::vector<std::byte> bytes(2 * s1 + s3 + s4);
    std::copy(reinterpret_cast<const std::byte *>(&detector_data.Resolution().x),
              reinterpret_cast<const std::byte *>(&detector_data.Resolution().x) + s1,
              &bytes[0]);

    std::copy(reinterpret_cast<const std::byte *>(&detector_data.Resolution().y),
              reinterpret_cast<const std::byte *>(&detector_data.Resolution().y) + s1,
              &bytes[0] + s1);

//    std::copy(reinterpret_cast<const char*>(&job_meta_information.ID()),
//              reinterpret_cast<const char*>(&job_meta_information.ID()) + s2,
//              bytes + 2 * s1);

    std::copy(reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]),
              reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]) + s3,
              &bytes[0] + 2 * s1);

    std::copy(reinterpret_cast<const std::byte *>(&sim_data.intensities[0]),
              reinterpret_cast<const std::byte *>(&sim_data.intensities[0]) + s4,
              &bytes[0] + 2 * s1 + s3);

    return bytes;
}
