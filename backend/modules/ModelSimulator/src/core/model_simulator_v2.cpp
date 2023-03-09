#include "core/model_simulator_v2.h"
#include "parameter_definitions/transformation_container.h"
#include "common/unitcell_utility.h"
#include "common/timer.h"
#include "spdlog/spdlog.h"
#include <thread>
#include "common/binary_serialization_utility.h"
#include <algorithm>

using json = nlohmann::json;

ModelSimulatorV2::ModelSimulatorV2(std::shared_ptr<HardwareInformation> hw_info)
        :
        hw_info_(hw_info) {
}

ModelSimulatorV2::~ModelSimulatorV2() {
    hw_info_->CleanUpDevices();
}


RequestResult ModelSimulatorV2::HandleRequest(const std::string &request) {
    try {
        Timer localTimer;
        Timer globalTimer;
        spdlog::info("Received request");

        globalTimer.Start();
        localTimer.Start();

        int message_size = *reinterpret_cast<int32_t const *>(&request[0]);
        std::string config(static_cast<const char *>(&request[sizeof(int32_t)]), message_size);
        int image_size = *reinterpret_cast<int32_t const *>(&request[sizeof(int) + message_size]);

        std::vector<std::byte> image_data(image_size);
        if (image_size > 0) {
            std::copy(reinterpret_cast<const std::byte *>(&request[2 * sizeof(int) + message_size]),
                      reinterpret_cast<const std::byte *>(&request[2 * sizeof(int)] + message_size + image_size),
                      &image_data[0]);
        }

        SimJob job = GisaxsTransformationContainer::CreateSimJobFromRequest(config);
        localTimer.End();
        spdlog::info("Preparing data took {} ms", localTimer.Duration());

        bool copy_intensities = true;
        std::shared_ptr<ImageData> img_data = nullptr;

        if (!image_data.empty()) {
            auto simulation_target_data = BinarySerializationUtility::ReadSimulationTargetData(image_data);
            img_data = std::make_shared<ImageData>(simulation_target_data);
            copy_intensities = false;
        }

        localTimer.Start();
        SimData sim_data = RunGISAXS(job, img_data, copy_intensities);

        localTimer.End();
        spdlog::info("Simulation took {} ms", localTimer.Duration());

        localTimer.Start();
        auto final_message = SerializeResult(sim_data, job.ExperimentInfo().DetectorConfig(), job.JobInfo(),
                                             copy_intensities);

        localTimer.End();
        spdlog::info("Serializing result took {} ms", localTimer.Duration());

        //hw_info_->CleanUpDevices();

        globalTimer.End();
        spdlog::info("Finished request in {} ms", globalTimer.Duration());
        return {final_message};
    }
    catch (const std::exception &e) {
        spdlog::error(e.what());
        return {std::vector<std::byte>()};
    }
}

SimData
ModelSimulatorV2::RunGISAXS(const SimJob &descr, std::shared_ptr<ImageData> real_img, bool copy_intensities) const {
    Device &device = hw_info_->LockAndReturnDevice();

    SimData sim_data = device.RunGISAXS(descr, real_img, copy_intensities);
    hw_info_->UnlockDevice(device);

    return sim_data;
}

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
    return "simulation";
}

std::vector<std::byte>
ModelSimulatorV2::SerializeResult(const SimData &sim_data, const DetectorConfiguration &detector_data,
                                  const JobMetaInformation &job_meta_information, bool copy_intensities) {
    if (!copy_intensities) {
        std::vector<std::byte> bytes(sizeof(MyType));
        auto byte_ptr = reinterpret_cast<const std::byte *>(&sim_data.fitness);
        std::copy(byte_ptr, byte_ptr + sizeof(MyType), &bytes[0]);
        return bytes;
    }

    std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> simulation_targets = job_meta_information.SimulationTargets();
    if (simulation_targets.empty()) {
        auto resolution = detector_data.Resolution();
        Vector2<int> target = {resolution.x - 1, resolution.y - 1};
        simulation_targets.emplace_back(
                GisaxsTransformationContainer::SimulationTargetDefinition{{0, 0}, target});
    }

    std::vector<std::byte> bytes = Serialize(simulation_targets, sim_data.normalized_intensities, sim_data.intensities,
                                             job_meta_information.Format());
    return bytes;
}


std::vector<std::byte> ModelSimulatorV2::Serialize(
        const std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> &target_definitions,
        const std::vector<unsigned char> &normalized_intensities,
        const std::vector<double> &intensities,
        IntensityFormat format) {
    size_t s1 = sizeof(int);
    size_t s2 = normalized_intensities.size() * sizeof normalized_intensities[0];
    size_t s3 = intensities.size() * sizeof intensities[0];

    size_t byte_count = 7 * target_definitions.size() * s1;
    if (format == IntensityFormat::double_precision) {
        byte_count += s3;
    }
    if (format == IntensityFormat::greyscale) {
        byte_count += s2;
    }


    std::vector<std::byte> bytes(byte_count);
    int current_offset = 0;
    for (const auto &st: target_definitions) {
        std::copy(reinterpret_cast<const std::byte *>(&st.start.x),
                  reinterpret_cast<const std::byte *>(&st.start.x) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        std::copy(reinterpret_cast<const std::byte *>(&st.start.y),
                  reinterpret_cast<const std::byte *>(&st.start.y) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        std::copy(reinterpret_cast<const std::byte *>(&st.end.x),
                  reinterpret_cast<const std::byte *>(&st.end.x) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        std::copy(reinterpret_cast<const std::byte *>(&st.end.y),
                  reinterpret_cast<const std::byte *>(&st.end.y) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        int x = st.end.x - st.start.x + 1;
        int y = st.end.y - st.start.y + 1;

        std::copy(reinterpret_cast<const std::byte *>(&x),
                  reinterpret_cast<const std::byte *>(&x) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        std::copy(reinterpret_cast<const std::byte *>(&y),
                  reinterpret_cast<const std::byte *>(&y) + s1,
                  &bytes[0] + current_offset);
        current_offset += s1;

        if (format == IntensityFormat::greyscale) {
            int single_value_size = sizeof normalized_intensities[0];

            std::copy(reinterpret_cast<const std::byte *>(&single_value_size),
                      reinterpret_cast<const std::byte *>(&single_value_size) + s1,
                      &bytes[0] + current_offset);
            current_offset += s1;

            size_t s2_local = x * y * single_value_size;
            std::copy(reinterpret_cast<const std::byte *>(&normalized_intensities[0]),
                      reinterpret_cast<const std::byte *>(&normalized_intensities[0]) + s2_local,
                      &bytes[0] + current_offset);
            current_offset += s2_local;
        }

        if (format == IntensityFormat::double_precision) {
            int single_value_size = sizeof intensities[0];

            std::copy(reinterpret_cast<const std::byte *>(&single_value_size),
                      reinterpret_cast<const std::byte *>(&single_value_size) + single_value_size,
                      &bytes[0] + current_offset);
            current_offset += s1;

            size_t s3_local = x * y * single_value_size;
            std::copy(reinterpret_cast<const std::byte *>(&intensities[0]),
                      reinterpret_cast<const std::byte *>(&intensities[0]) + s3_local,
                      &bytes[0] + current_offset);
            current_offset += s3_local;
        }
    }
    return bytes;
}
