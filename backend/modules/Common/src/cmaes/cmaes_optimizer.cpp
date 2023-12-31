//
// Created by phili on 23.10.2022.
//

#include "cmaes/cmaes_optimizer.h"
#include <iostream>

CmaesOptimizer::CmaesOptimizer(
        std::function<std::vector<double>(const std::vector<std::vector<double>> &dv)> function,
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

FitData CmaesOptimizer::Optimize() {

    std::vector<double> fitness_value_per_iteration;

    for (int i = 0; i < max_iterations_; ++i) {
        auto solutions = std::vector<Solution>();

        std::vector<std::vector<double>> vectors(cma_.PopulationSize());
        std::vector<Eigen::VectorX<double>> parameters(cma_.PopulationSize());
        for (int j = 0; j < cma_.PopulationSize(); j++) {
            Eigen::VectorX<double> parameterVector = cma_.Ask();
            parameters.at(j) = parameterVector;
            vectors.at(j) = std::vector<double>(parameterVector.data(),
                                                parameterVector.data() + parameterVector.size());
        }

        std::vector<double> fitness = function_(vectors);

        for (int j = 0; j < cma_.PopulationSize(); j++) {
            solutions.emplace_back(parameters.at(j), fitness.at(j));
        }

        cma_.Tell(solutions);

        const auto best_solution_it = std::min_element(std::begin(solutions), std::end(solutions));
        fitness_value_per_iteration.emplace_back(best_solution_it->Fitness());

//        cma_.ShouldStop()
        if (cma_.ShouldStop()) {
            std::vector<double> final_parameters;
            std::shared_ptr<Solution> best_solution = std::make_shared<Solution>(*best_solution_it);
            if (best_solution != nullptr) {
                std::cout << "Fitness: " << (*best_solution_it).Fitness() << " Params: ";
                std::string params;
                for (auto parameter: (*best_solution_it).Parameters()) {
                    final_parameters.emplace_back(parameter);
                    params = params + " " + std::to_string(parameter);
                }
                std::cout << params << std::endl;
            }
            return {fitness_value_per_iteration, final_parameters};
        }
    }
}
