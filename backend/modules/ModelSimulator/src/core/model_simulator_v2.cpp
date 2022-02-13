#include "core/model_simulator_v2.h"
#include "parameter_definitions/transformation_container.h"

#include <iostream>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include <chrono>

using namespace distribution_functions;
using json = nlohmann::json;

ModelSimulatorV2::ModelSimulatorV2()
        :
        hw_info_(std::make_shared<HardwareInformation>()) {
}

ModelSimulatorV2::~ModelSimulatorV2() {
    hw_info_->CleanUpDevices();
}


std::string ModelSimulatorV2::HandleRequest(const std::string &request) const {

    //return request;

    json data = json::parse(request);

    json detector = data.at("config").at("detector");
    json shapes = data.at("config").at("shapes");
    json sample = data.at("config").at("sample");
    json beam = data.at("config").at("beam");
    json unitcellMeta = data.at("config").at("unitcellMeta");

    std::cout << "Got request..." << std::endl;

    auto detectorSetupContainer = detector.get<DetectorSetupContainer>();
    auto shapesContainer = shapes.get<ShapeContainer>();
    auto sampleContainer = sample.get<SampleContainer>();
    auto beamContainer = beam.get<BeamContainer>();
    auto unitcellMetaContainer = unitcellMeta.get<UnitcellMetaContainer>();

    std::shared_ptr<UnitcellV2> unitcell = std::make_shared<UnitcellV2>(std::move(shapesContainer.shapes),
                                                                        unitcellMetaContainer.repetitions,
                                                                        unitcellMetaContainer.translation);
    DetectorSetup detectorSetup(detectorSetupContainer.pixelsize, detectorSetupContainer.sampleDistance,
                                detectorSetupContainer.beamImpact, detectorSetupContainer.resolution);

    MyType wavelength = (MyType) 1239.84 / beamContainer.photonEv;
    MyType alphaiInRad = (MyType) beamContainer.alphai * (MyType) 0.017453;
    BeamConfiguration beamConfig(alphaiInRad, detectorSetup.Directbeam(), wavelength, 0.1);
    Sample sampleConfig(Layer(sampleContainer.substrate_delta, sampleContainer.substrate_beta, -1, 0),
                        sampleContainer.layers);

    Timer t;
    t.Start();
    std::shared_ptr<ExperimentalModel> experimentalModel = std::make_shared<ExperimentalModel>(detectorSetup,
                                                                                               std::vector<int>{},
                                                                                               beamConfig, sampleConfig,
                                                                                               detectorSetup.SampleDistance(),
                                                                                               0);
    t.End();

    std::cout << t.Duration() << " s" << std::endl;

    SimJob job("000101", unitcell, experimentalModel, false);

    SimData sim_data = RunGISAXS(job, nullptr, true);

    auto size = sim_data.normalized_intensities.size();

    nlohmann::json message;
    message["intensities"] = sim_data.normalized_intensities;
    message["width"] = detectorSetup.Resolution().x;
    message["height"] = detectorSetup.Resolution().y;
    message["id"] = job.Uuid();

    auto final_message = message.dump();
    hw_info_->CleanUpDevices();

    std::cout << "Finished request..." << std::endl;


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
