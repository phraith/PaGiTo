#include "core/model_simulator_v2.h"
#include "parameter_definitions/transformation_container.h"
#include "common/unitcell_utility.h"
#include "common/timer.h"
#include "spdlog/spdlog.h"
<<<<<<< Updated upstream
#include <thread>
=======
#include "common/binary_serialization_utility.h"
>>>>>>> Stashed changes
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

        bool copy_intensities = true;
        std::shared_ptr<ImageData> img_data = nullptr;

        if (!image_data.empty())
        {
            auto simulation_target_data = BinarySerializationUtility::ReadSimulationTargetData(image_data);
            img_data = std::make_shared<ImageData>(simulation_target_data);
            copy_intensities = false;
        }


        localTimer.Start();
        SimData sim_data = RunGISAXS(job, img_data, copy_intensities);
        localTimer.End();
        //spdlog::info("Simulation took {} ms", localTimer.Duration());

    localTimer.Start();

        auto final_message = CreateSerializedByteResult(sim_data, job.ExperimentInfo().DetectorConfig(), job.JobInfo(), copy_intensities);
        localTimer.End();
        //spdlog::info("Serializing result took {} ms", localTimer.Duration());

    //hw_info_->CleanUpDevices();

    globalTimer.End();
    spdlog::info("Finished request in {} ms", globalTimer.Duration());
    return final_message;
}

SimData ModelSimulatorV2::RunGISAXS(const SimJob &descr, std::shared_ptr<ImageData> real_img, bool copy_intensities) const {
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
<<<<<<< Updated upstream
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
=======
                                             const JobMetaInformation &job_meta_information, bool copy_intensities) {
    if (!copy_intensities)
    {
        std::vector<std::byte> bytes(sizeof(MyType));
        auto byte_ptr = reinterpret_cast<const std::byte *>(&sim_data.fitness);
        std::copy(byte_ptr, byte_ptr + sizeof(MyType), &bytes[0]);
        return bytes;
    }

    if (job_meta_information.SimulationTargets().size() == 0) {
        size_t s1 = 2 * sizeof detector_data.Resolution().x;
        size_t s2 = sim_data.normalized_intensities.size() * sizeof sim_data.normalized_intensities[0];
        size_t s3 = sim_data.intensities.size() * sizeof(double);

        std::vector<std::byte> bytes(s1 + s2 + s3);
        std::copy(reinterpret_cast<const std::byte *>(&detector_data.Resolution().x),
                  reinterpret_cast<const std::byte *>(&detector_data.Resolution().x) + s1,
                  &bytes[0]);

        std::copy(reinterpret_cast<const std::byte *>(&detector_data.Resolution().y),
                  reinterpret_cast<const std::byte *>(&detector_data.Resolution().y) + s1,
                  &bytes[0] + s1);

        if (job_meta_information.Format() == IntensityFormat::greyscale) {
            std::copy(reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]),
                      reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]) + s2,
                      &bytes[0] + 2 * s1);
        }

        if (job_meta_information.Format() == IntensityFormat::double_precision) {
            std::copy(reinterpret_cast<const std::byte *>(&sim_data.intensities[0]),
                      reinterpret_cast<const std::byte *>(&sim_data.intensities[0]) + s3,
                      &bytes[0] + 2 * s1);
        }
        return bytes;

    } else {
        size_t s1 = sizeof(int);
        size_t s2 = sim_data.normalized_intensities.size() * sizeof sim_data.normalized_intensities[0];
        size_t s3 = sim_data.intensities.size() * sizeof(double);

        std::vector<std::byte> bytes(2 * job_meta_information.SimulationTargets().size() * s1 + s2 + s3);
        int current_offset = 0;
        for (const auto &st: job_meta_information.SimulationTargets()) {
            int x = st.end.x - st.start.x + 1;
            int y = st.start.y - st.end.y + 1;

            size_t s2_local = x * y * sizeof sim_data.normalized_intensities[0];
            size_t s3_local = x * y * sizeof sim_data.intensities[0];

            std::copy(reinterpret_cast<const std::byte *>(&x),
                      reinterpret_cast<const std::byte *>(&x) + s1,
                      &bytes[0] + current_offset);

            std::copy(reinterpret_cast<const std::byte *>(&y),
                      reinterpret_cast<const std::byte *>(&y) + s1,
                      &bytes[0] + current_offset + s1);

            if (job_meta_information.Format() == IntensityFormat::greyscale) {
                std::copy(reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]),
                          reinterpret_cast<const std::byte *>(&sim_data.normalized_intensities[0]) + s2_local,
                          &bytes[0] + current_offset + 2 * s1);
            }

            if (job_meta_information.Format() == IntensityFormat::double_precision) {
                std::copy(reinterpret_cast<const std::byte *>(&sim_data.intensities[0]),
                          reinterpret_cast<const std::byte *>(&sim_data.intensities[0]) + s3_local,
                          &bytes[0] + current_offset + 2 * s1);
            }
        }
        return bytes;
    }
>>>>>>> Stashed changes
}
