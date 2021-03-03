#ifndef MODEL_FITTER_CORE_PUBLISHER_H
#define MODEL_FITTER_CORE_PUBLISHER_H

#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "core/fitting_result.h"

class Publisher {
public:
	Publisher(const std::string ip);
	~Publisher();

	void InsertFittingResult(std::shared_ptr<FittingResult> result);

private:
	void Publish();
	std::shared_ptr<FittingResult> TakeFittingResult();

	const std::string ip_;
	std::thread publication_handler_;

	std::deque <std::shared_ptr<FittingResult>> fitting_results_;
	std::mutex result_mutex_;
	std::condition_variable result_cv_;

	bool quit_work_;
};

#endif