//#ifndef MODEL_FITTER_CORE_CONNECTOR_H
//#define MODEL_FITTER_CORE_CONNECTOR_H
//
//#include <string>
//#include <thread>
//#include <deque>
//#include <mutex>
//#include <condition_variable>
//#include <core/fitting_description.h>
//
//#include "serialized_fitting_description.capnp.h"
//
//class Connector
//{
//public:
//	Connector(const std::string ip);
//	~Connector();
//
//	void InsertFittingJob(std::shared_ptr<FitJob> fitting_job);
//	std::shared_ptr<FitJob> TakeFittingJob();
//private:
//	void Listen();
//	std::shared_ptr<FitJob> ConstructFittingDescription(const SerializedFittingDescription::Reader& data);
//
//	const std::string ip_;
//	std::thread connection_handler_;
//
//	std::deque<std::shared_ptr<FitJob>> fitting_jobs_;
//	std::mutex job_mutex_;
//	std::condition_variable job_cv_;
//
//	bool quit_work_;
//};
//
//#endif