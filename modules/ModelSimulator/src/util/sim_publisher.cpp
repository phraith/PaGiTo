#include "util/sim_publisher.h"

#include <iostream>

#include <zmq.hpp>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include <serialized_simulation_description.capnp.h>
#include <capnp/serialize.h>

SimPublisher::SimPublisher()
{
}

SimPublisher::SimPublisher(const std::string ip)
	:
	ip_(ip),
	publication_handler_(&SimPublisher::Publish, this),
	quit_work_(false)
{
}

SimPublisher::~SimPublisher()
{
	if (publication_handler_.joinable())
		publication_handler_.join();
}

void SimPublisher::InsertSimResult(std::shared_ptr<SimResult> result)
{
	std::lock_guard <std::mutex> lock(result_mutex_);
	sim_results_.emplace_back(result);

	result_cv_.notify_one();
}

void SimPublisher::Publish()
{
	zmq::context_t context;
	zmq::socket_t socket(context, zmq::socket_type::pub);
	socket.bind("tcp://" + ip_);

	std::cout << "tcp://" + ip_ + "\n" << std::endl;

	while (std::shared_ptr<SimResult> result = TakeSimResult())
	{

		::capnp::MallocMessageBuilder message;

		SerializedSimResult::Builder sfr = message.initRoot<SerializedSimResult>();
		sfr.setClientId(result->Uuid());

		sfr.setXDim(result->Data().resolution.x);
		sfr.setYDim(result->Data().resolution.y);

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
			device_timings[i].setKernelTime(device_timing.kernel_time);
			device_timings[i].setSimulationTime(device_timing.full_runtime);
		}

		auto result_m = capnp::messageToFlatArray(message);

		try
		{
			socket.send(zmq::buffer(result->Uuid()), zmq::send_flags::sndmore);
			socket.send(zmq::buffer(result_m.asBytes().begin(), result_m.asBytes().size()));
			std::cout << "send result" << std::endl;
		}
		catch (zmq::error_t& e) {
			std::cout << e.what() << std::endl;
		}
	}
	std::cout << "unbinding socket.." << std::endl;
	socket.unbind("tcp://" + ip_);
}

std::shared_ptr<SimResult> SimPublisher::TakeSimResult()
{
	auto lk = std::unique_lock<std::mutex>(result_mutex_);
	result_cv_.wait(lk, [&]
		{
			return !sim_results_.empty() || quit_work_;
		});

	if (sim_results_.empty())
		return nullptr;

	auto sim_result = sim_results_.front();
	sim_results_.pop_front();

	if (sim_result->IsLast())
		quit_work_ = true;

	return sim_result;
}
