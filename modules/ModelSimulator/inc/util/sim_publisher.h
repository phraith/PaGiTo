#ifndef MODEL_SIMULATOR_UTIL_PUBLISHER_H
#define MODEL_SIMULATOR_UTIL_PUBLISHER_H

#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "util/sim_result.h"

class SimPublisher {
public:
	SimPublisher();
	SimPublisher(const std::string ip);
	~SimPublisher();

	void InsertSimResult(std::shared_ptr<SimResult> result);

private:
	void Publish();
	std::shared_ptr<SimResult> TakeSimResult();

	const std::string ip_;
	std::thread publication_handler_;

	std::deque <std::shared_ptr<SimResult>> sim_results_;
	std::mutex result_mutex_;
	std::condition_variable result_cv_;

	bool quit_work_;
};

#endif