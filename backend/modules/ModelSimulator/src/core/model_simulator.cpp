#include "core/model_simulator.h"

#include <iostream>
#include <thread>
#include <algorithm>
#include <stdexcept>

#include <cuda.h>

#include "vector_types.h"

#include <chrono>

using namespace distribution_functions;

ModelSimulator::ModelSimulator()
	:
	hw_info_(std::make_shared<HardwareInformation>()),
	quit_work_(false)
{
}

ModelSimulator::ModelSimulator(const std::string& connector_port, const std::string& publisher_port)
	:
	hw_info_(std::make_shared<HardwareInformation>()),
	connector_port_(connector_port),
	publisher_port_(publisher_port),
	job_provider_("0.0.0.0:" + connector_port),
	result_publisher_("0.0.0.0:" + publisher_port),
	quit_work_(false)
{
}

ModelSimulator::~ModelSimulator()
{
	hw_info_->CleanUpDevices();
}

SimData ModelSimulator::RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities) const
{
	Device &device = LockAndReturnDevice();
	
	SimData sim_data = device.RunGISAXS(descr, real_img, copy_intensities);
	
	UnlockDevice(device);

	cv_.notify_one();
	return sim_data;
}

void ModelSimulator::Run()
{
	if (connector_port_ == "" || publisher_port_ == "")
		throw std::runtime_error("Connector or Publisher are not properly set up!");

	while (std::shared_ptr<SimJob> job = job_provider_.TakeSimulationJob())
	{
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		SimData sim_data = RunGISAXS(*job, nullptr, true);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		result_publisher_.InsertSimResult(std::make_shared<SimResult>(job->Uuid(), job->IsLast(), sim_data, GetDeviceTimings()));
		for (auto& timing : GetDeviceTimings())
		{
			std::cout << "Device Name: " << timing.device_name << std::endl;
			std::cout << "Average runtime of simulation: " << timing.average_full_runtime << std::endl;
			std::cout << "Added up runtime of simulation: " << timing.full_runtime << std::endl;
			std::cout << "Average runtime of kernel: " << timing.average_kernel_runtime << std::endl;
			std::cout << "Added up runtime of kernel: " << timing.kernel_time << std::endl;
			std::cout << "Complete runs of simulation: " << timing.runs << std::endl << std::endl;
		}

		hw_info_->CleanUpDevices();
		
		std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	}
}

void ModelSimulator::ResetTimers()
{
	hw_info_->ResetTimers();
}

void ModelSimulator::Reset()
{
	hw_info_->CleanUpDevices();
}

Device& ModelSimulator::LockAndReturnDevice() const
{
	auto lk = std::unique_lock<std::mutex>(mutex_);

	Device* device = nullptr;

	cv_.wait(lk, [&]
		{
			device = hw_info_->FindFreeDevice();
			return device != nullptr;
		});

	device->SetStatus(WorkStatus::kWorking);

	return *device;
}

void ModelSimulator::UnlockDevice(Device& device) const
{
	auto lk = std::unique_lock<std::mutex>(mutex_);
	device.SetStatus(WorkStatus::kIdle);
}

const std::vector<SimulationInterval> ModelSimulator::Distribute(std::shared_ptr<SimJob> simDescr, DistributionType distType, GpuDevice& device, int blocksize) const
{
	try
	{
		return dist_func_map.at(distType) (simDescr, device, blocksize);
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
	return std::vector<SimulationInterval>();
}

std::vector<TimeMeasurement> ModelSimulator::GetDeviceTimings() const
{
	std::vector<TimeMeasurement> timings;
	for (const auto& device : hw_info_->DeviceInfo())
	{
		if (device->KernelTime() != 0 && device->FullTime() != 0)
			timings.emplace_back(TimeMeasurement{device->Name(), device->KernelTime() / 1000.f, device->FullTime(), device->AverageKernelTime() / 1000.f, device->AverageFullTime(), device->Runs() } );
	}

	return timings;
}
