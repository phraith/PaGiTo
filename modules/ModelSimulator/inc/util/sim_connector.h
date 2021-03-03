#ifndef MODEL_SIMULATOR_UTIL_CONNECTOR_H
#define MODEL_SIMULATOR_UTIL_CONNECTOR_H

#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <common/simulation_description.h>

#include "serialized_simulation_description.capnp.h"

class SimConnector
{
public:
	SimConnector();
	SimConnector(const std::string ip);
	~SimConnector();

	void InsertSimulationJob(std::shared_ptr<SimJob> fitting_job);
	std::shared_ptr<SimJob> TakeSimulationJob();
private:
	void Listen();
	std::shared_ptr<SimJob> ConstructSimulationDescription(const SerializedSimulationDescription::Reader& data);

	const std::string ip_;
	std::thread connection_handler_;

	std::deque<std::shared_ptr<SimJob>> simulation_jobs_;
	std::mutex job_mutex_;
	std::condition_variable job_cv_;

	bool quit_work_;
};

#endif