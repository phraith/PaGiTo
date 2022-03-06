#include "core/model_simulator_v2.h"
#include "parameter_definitions/transformation_container.h"
#include "common/unitcell_utility.h"
#include "common/timer.h"

#include <iostream>
#include <thread>
#include <algorithm>

using namespace GisaxsTransformationContainer;
using json = nlohmann::json;

ModelSimulatorV2::ModelSimulatorV2()
        :
        hw_info_(std::make_shared<HardwareInformation>()) {
}

ModelSimulatorV2::~ModelSimulatorV2() {
    hw_info_->CleanUpDevices();
}


std::string ModelSimulatorV2::HandleRequest(const std::string &request) const {
    json data = json::parse(request);

    json detector = data.at("detector");
    json shapes = data.at("shapes");
    json sample = data.at("sample");
    json beam = data.at("beam");
    json unitcellMeta = data.at("unitcellMeta");

    std::cout << "Got request..." << std::endl;

    auto detector_container = ConvertToDetector(detector);
    auto shapes_container = ConvertToFlatShapes(shapes);
    auto sample_container = ConvertToSample(sample);
    auto beam_container = ConvertToBeam(beam);
    auto unitcell_meta_container = ConvertToUnitcellMeta(unitcellMeta);
    auto flat_unitcell = FlatUnitcellV2(shapes_container, unitcell_meta_container.repetitions,
                                        unitcell_meta_container.translation);

    DetectorConfiguration detector_config(detector_container);
    BeamConfiguration beam_config(beam_container.alphai, detector_config.Directbeam(), beam_container.photonEv, 0.1);
    SampleConfiguration sample_config(Layer(sample_container.substrate_delta, sample_container.substrate_beta, -1, 0),
                                      sample_container.layers);

    SimJob job(JobMetaInformation{"1"},
               ExperimentalData{detector_config, beam_config, sample_config, flat_unitcell});
    Timer t;
    t.Start();
    SimData sim_data = RunGISAXS(job, nullptr, true);
    t.End();

    auto final_message = CreateSerializedResult(sim_data, detector_config, job.JobInfo());
    hw_info_->CleanUpDevices();

    std::cout << "Finished request in " << t.Duration() << "ms" << std::endl;
    return final_message;
}

SimData ModelSimulatorV2::RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities) const {
    Device &device = LockAndReturnDevice();

    SimData sim_data = device.RunGISAXS(descr, real_img, copy_intensities);

    UnlockDevice(device);

    cv_.notify_one();
    return sim_data;
}

Device &ModelSimulatorV2::LockAndReturnDevice() const {
    auto lk = std::unique_lock<std::mutex>(mutex_);

    Device *device = nullptr;

    cv_.wait(lk, [&] {
        device = hw_info_->FindFreeDevice();
        return device != nullptr;
    });

    device->SetStatus(WorkStatus::kWorking);

    return *device;
}

void ModelSimulatorV2::UnlockDevice(Device &device) const {
    auto lk = std::unique_lock<std::mutex>(mutex_);
    device.SetStatus(WorkStatus::kIdle);
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
