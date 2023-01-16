//
// Created by phili on 23.10.2022.
//

#include "cmaes/cmaes_optimizer.h"
#include <iostream>

CmaesOptimizer::CmaesOptimizer(std::function<double(const std::vector<double> &dv)> function,
                               const std::vector<double> &initial, const std::vector<double> &lower,
                               const std::vector<double> &upper,
                               double sigma, int max_iterations, int rand_seed)
        :
        function_(function),
        initial_(initial),
        rand_seed_(rand_seed),
        cma_(initial, sigma, upper, lower),
        max_iterations_(max_iterations) {

}

std::shared_ptr<Solution> CmaesOptimizer::Optimize() {
    for (int i = 0; i < max_iterations_; ++i) {
        auto solutions = std::vector<Solution>();
        for (int j = 0; j < cma_.PopulationSize(); j++) {
            Eigen::VectorX<double> parameterVector = cma_.Ask();
            std::vector<double> parameters(parameterVector.data(), parameterVector.data() + parameterVector.size());
            double fitness = function_(parameters);
            solutions.emplace_back(parameterVector, fitness);
        }

        cma_.Tell(solutions);
        if (cma_.ShouldStop()) {

            const auto best_solution_it = std::min_element(std::begin(solutions), std::end(solutions));
            std::shared_ptr<Solution> best_solution = std::make_shared<Solution>(*best_solution_it);

            if (best_solution != nullptr) {
                std::cout << "Fitness: " << (*best_solution_it).Fitness() << " Params: ";
                std::string params;
                for (auto parameter: (*best_solution_it).Parameters()) {
                    params = params + " " + std::to_string(parameter);
                }
                std::cout << params << std::endl;
            }
            return best_solution;
        }
    }
}