#ifndef MODEL_SIMULATOR_UTIL_CONNECTOR_H
#define MODEL_SIMULATOR_UTIL_CONNECTOR_H

#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <common/simulation_description.h>

#include "serialized_simulation_description.capnp.h"
#include <nlohmann/json.hpp>

class SimDataContainer
{

public:
	SimDataContainer();
	SimDataContainer(nlohmann::json& scattering, nlohmann::json& detector, nlohmann::json& substrate_refindex);

	nlohmann::json scattering;
	nlohmann::json detector;
	nlohmann::json substrate_refindex;
};

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
	bool ExperimentalSetupHasChanged(const nlohmann::json& detector_data, const nlohmann::json& scattering, const nlohmann::json& substrate_refindex);
	const std::string ip_;
	std::thread connection_handler_;

	std::deque<std::shared_ptr<SimJob>> simulation_jobs_;
	std::mutex job_mutex_;
	std::condition_variable job_cv_;

	SimDataContainer currentDataContainer_;
	std::shared_ptr<ExperimentalModel> current_model_;

	bool quit_work_;
};

#endif