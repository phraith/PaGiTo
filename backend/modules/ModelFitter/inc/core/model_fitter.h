#ifndef MODEL_FITTER_CORE_MODEL_FITTER_H
#define MODEL_FITTER_CORE_MODEL_FITTER_H

#include "core/fitting_description.h"

#include <mutex>

#include "core/model_simulator.h"
#include "core/connector.h"
#include "core/publisher.h"

#include "common/timer.h"
#include "common/unitcell.h"

class ModelFitter
{
public:
	ModelFitter();

	void Run();
	
private:
	std::shared_ptr<ModelSimulator> ms_;

	Connector job_provider_;
	Publisher result_publisher_;

	bool quit_work_;

	Timer fitting_timer_;
};

#endif