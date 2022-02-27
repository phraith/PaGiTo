#include "core/publisher.h"

#include <iostream>

#include <zmq.hpp>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include <serialized_fitting_description.capnp.h>
#include <capnp/serialize.h>

#include <chrono>

Publisher::Publisher(const std::string ip)
	:
	ip_(ip),
	publication_handler_(&Publisher::Publish, this),
	quit_work_(false)
{
}

Publisher::~Publisher()
{
	publication_handler_.join();
}

void Publisher::InsertFittingResult(std::shared_ptr<FittingResult> result)
{
	std::lock_guard <std::mutex> lock(result_mutex_);
	fitting_results_.emplace_back(result);

	result_cv_.notify_one();
}

void Publisher::Publish()
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::pub);
    socket.bind("tcp://" + ip_);

	std::cout << "tcp://" + ip_ + "\n" << std::endl;

    while (std::shared_ptr<FittingResult> result = TakeFittingResult())
	{

		::capnp::MallocMessageBuilder message;

		SerializedFittingResult::Builder sfr = message.initRoot<SerializedFittingResult>();
		sfr.setClientId(result->Uuid());
		sfr.setFitness(result->Data().fitness);
		sfr.setScale(result->Data().scale);
		auto fitted_shapes = result->FittedShapes();
		::capnp::List<SerializedFittingResult::FittedShape>::Builder sfrShapes = sfr.initFittedShapes(fitted_shapes.size());
		for (int i = 0; i < result->FittedShapes().size(); ++i)
		{
			auto& current_shape = fitted_shapes[i];
			sfrShapes[i].setType(current_shape.type);
			::capnp::List<SerializedFittingResult::FittedParameter>::Builder sfrParams = sfrShapes[i].initParameters(current_shape.parameters.size());
			for (int j = 0; j < current_shape.parameters.size(); ++j)
			{
				auto& current_param = current_shape.parameters[j];
				sfrParams[j].setValue(current_param.value);
				sfrParams[j].setStddev(current_param.stddev);
				sfrParams[j].setType(current_param.type);
			}
		}

		auto sim_int = sfr.initSimulatedIntensities(result->Data().intensities.size());
		auto sim_qx = sfr.initSimulatedQx(result->Data().qx.size());
		auto sim_qy = sfr.initSimulatedQy(result->Data().qy.size());
		auto sim_qz = sfr.initSimulatedQz(result->Data().qz.size());
		for (int i = 0; i < result->Data().intensities.size(); ++i)
		{
			sim_int.set(i, result->Data().intensities.at(i));

			sim_qx.set(i, result->Data().qx.at(i));
			sim_qy.set(i, result->Data().qy.at(i));
			sim_qz.set(i, result->Data().qz.at(i));
		}

		auto device_timings = sfr.initDeviceTimingData(result->DeviceTimings().size());
		for (int i = 0; i < result->DeviceTimings().size(); ++i)
		{
			auto& device_timing = result->DeviceTimings().at(i);
			device_timings[i].setDeviceName(device_timing.device_name);
			device_timings[i].setAverageKernelTime(device_timing.average_kernel_runtime);
			device_timings[i].setKernelTime(device_timing.kernel_time);

			device_timings[i].setAverageSimulationTime(device_timing.average_full_runtime);
			device_timings[i].setSimulationTime(device_timing.full_runtime);

			device_timings[i].setSimRuns(device_timing.runs);
			device_timings[i].setFittingTime(result->FittingTime());
		}

		auto result_m = capnp::messageToFlatArray(message);

		try 
		{
			socket.send(zmq::buffer(result->Uuid()), zmq::send_flags::sndmore);
			socket.send(zmq::buffer(result_m.asBytes().begin(), result_m.asBytes().size()));
			std::cout << "Send result" << std::endl;
		}
		catch (zmq::error_t & e) {
			std::cout << e.what() << std::endl;
		}
    }
	std::cout << "unbinding socket.." << std::endl;
    socket.unbind("tcp://" + ip_);
}

std::shared_ptr<FittingResult> Publisher::TakeFittingResult()
{
	auto lk = std::unique_lock<std::mutex>(result_mutex_);
	result_cv_.wait(lk, [&]
		{
			return !fitting_results_.empty() || quit_work_;
		});

	if (fitting_results_.empty())
		return nullptr;

	auto fitting_result = fitting_results_.front();
	fitting_results_.pop_front();

	if (fitting_result->IsLast())
		quit_work_ = true;

	return fitting_result;
}