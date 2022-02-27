//#include "core/model_fitter.h"
//
//#include "assert.h"
//
//#include <pagmo/algorithm.hpp>
//#include <pagmo/algorithms/sade.hpp>
//#include <pagmo/algorithms/cmaes.hpp>
//#include <pagmo/archipelago.hpp>
//#include <pagmo/problem.hpp>
//#include <pagmo/problems/schwefel.hpp>
//#include <pagmo/islands/thread_island.hpp>
//#include <pagmo/algorithms/de.hpp>
//#include <pagmo/algorithms/gaco.hpp>
//#include <pagmo/algorithms/ihs.hpp>
//#include <pagmo/algorithms/pso_gen.hpp>
//#include <pagmo/problems/rosenbrock.hpp>
//#include <pagmo/problems/cec2006.hpp>
//
//#include "core/fitting_result.h"
//#include "common/standard_defs.h"
//
//using namespace pagmo;
//
//struct gisaxs {
//
//	/*gisaxs()
//		:
//		fit_job_(nullptr),
//		ms_(nullptr)
//	{}*/
//
//	gisaxs(std::shared_ptr<FitJob> fit_job = nullptr, std::shared_ptr<ModelSimulator> ms = nullptr)
//		:
//		fit_job_(fit_job),
//		ms_(ms)
//
//
//	{
//		if (fit_job_ == nullptr || ms_ == nullptr)
//		{
//			pagmo_throw(std::invalid_argument,
//				"Gisaxs Function must initilize the Model and ModelSimulator!");
//		}
//	}
//
//	// Implementation of the objective function.
//	vector_double fitness(const vector_double& dv) const
//	{
//		vector_double fitness_eval(1, 0);
//		std::shared_ptr<Unitcell> unitcell = std::make_shared<Unitcell>(*(fit_job_->HUnitcell()));
//		unitcell->UpdateRvs(std::vector<MyType> {dv.begin(), dv.end()});
//		SimData sim_data = ms_->RunGISAXS(SimJob{ fit_job_->Uuid(), unitcell, fit_job_->GetModel(), fit_job_->IsLast() }, fit_job_->RealImg());
//		fitness_eval[0] = sim_data.fitness;
//
//		return fitness_eval;
//	}
//
//	// Implementation of the box bounds.
//	std::pair<vector_double, vector_double> get_bounds() const
//	{
//		return { fit_job_->HUnitcell()->LeftBounds(), fit_job_->HUnitcell()->RightBounds() };
//	}
//
//	std::shared_ptr<FitJob> fit_job_;
//	std::shared_ptr<ModelSimulator> ms_;
//};
//
//ModelFitter::ModelFitter()
//	:
//	ms_(std::make_shared<ModelSimulator>()),
//	job_provider_("0.0.0.0:5556"),
//	result_publisher_("0.0.0.0:5557"),
//	quit_work_(false)
//{
//}
//
//void ModelFitter::Run()
//{
//	while (std::shared_ptr<FitJob> job = job_provider_.TakeFittingJob())
//	{
//		fitting_timer_.Start();
//
//		archipelago archi{ job->Populations(), thread_island(false), cmaes {2, -1,  -1,  -1, -1, 0.5, 1e-18, 1e-18, false, true},  problem{ gisaxs(job, ms_) }, job->Individuals() };
//		ms_->ResetTimers();
//		archi.evolve(job->Evolutions());
//		archi.wait_check();
//		fitting_timer_.End();
//		MyType fitting_time = fitting_timer_.Duration();
//		std::cout << "Complete fitting time: " << fitting_time << std::endl;
//
//
//
//		for (auto& timing : ms_->GetDeviceTimings())
//		{
//			if (timing.runs == 100)
//				std::cout << "stop!" << std::endl;
//
//			std::cout << "Device Name: " << timing.device_name << std::endl;
//			std::cout << "Average runtime of simulation: " << std::setw(9) << timing.average_full_runtime << std::endl;
//			std::cout << "Added up runtime of simulation: " << std::setw(9) << timing.full_runtime << std::endl;
//			std::cout << "Average runtime of kernel: " << std::setw(9) << timing.average_kernel_runtime << std::endl;
//			std::cout << "Added up runtime of kernel: " << std::setw(9) << timing.kernel_time << std::endl;
//			std::cout << "Complete runs of simulation: " << timing.runs << std::endl << std::endl;
//		}
//
//
//		std::cout << "champion_fitness..." << std::endl;
//		double current_min = std::numeric_limits<MyType>::max();
//		int min_index = -1;
//
//
//		auto champions = archi.get_champions_f();
//		for (int i = 0; i < champions.size(); ++i)
//		{
//			if (champions.at(i)[0] < current_min)
//			{
//				current_min = champions.at(i)[0];
//				min_index = i;
//			}
//		}
//
//		assert(min_index != -1);
//		auto ff = archi.get_champions_x();
//		auto gg = archi.get_champions_f();
//
//		for (int i = 0; i < archi.get_champions_x()[min_index].size(); ++i)
//		{
//			std::cout << archi.get_champions_x()[min_index][i] << std::endl;
//		}
//		std::cout << archi.get_champions_f()[min_index][0] << std::endl;
//
//		auto best_parameters = archi.get_champions_x().at(min_index);
//		std::vector<FittedParameter> fitted_params;
//
//		job->HUnitcell()->UpdateRvs(std::vector<MyType> {best_parameters.begin(), best_parameters.end()});
//		auto device_timings = ms_->GetDeviceTimings();
//
//		SimData sim_data = ms_->RunGISAXS(SimJob{ job->Uuid(), job->HUnitcell() , job->GetModel(), job->IsLast()}, job->RealImg(), true);
//
//		result_publisher_.InsertFittingResult(std::make_shared<FittingResult>(job->Uuid(), job->IsLast(), job->HUnitcell()->FittedShapes(), sim_data, device_timings, fitting_time));
//
//		ms_->Reset();
//	}
//}