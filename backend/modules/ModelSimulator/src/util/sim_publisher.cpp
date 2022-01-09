#include "util/sim_publisher.h"
#include <iostream>
#include <zmq.hpp>
#include <serialized_simulation_description.capnp.h>
#include <nlohmann/json.hpp>

SimPublisher::SimPublisher()
	:
	quit_work_(false)
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
		std::string message_string = CreateMessageString(result);
		try
		{
			const std::vector<unsigned char>& f = std::vector<unsigned char>(result->Data().normalized_intensities);
			socket.send(zmq::buffer(result->Uuid()), zmq::send_flags::sndmore);
			//socket.send(zmq::buffer(message_string.data(), message_string.size()));
			socket.send(zmq::buffer(&f[0], f.size()));

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

std::string SimPublisher::CreateMessageString(std::shared_ptr<SimResult> result)
{
	nlohmann::json message;
	const std::vector<unsigned char>& f = std::vector<unsigned char>(result->Data().normalized_intensities);

	message["intensities"] = f;
	message["id"] = result->Uuid();
	return message.dump();
}
