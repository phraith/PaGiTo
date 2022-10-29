//
// Created by phili on 23.10.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_CMAES_OPTIMIZER_H
#define GISAXSMODELINGFRAMEWORK_CMAES_OPTIMIZER_H

#include "cmaes.h"

class CmaesOptimizer {
public:
    CmaesOptimizer(double (*function)(const std::vector<double> &), const std::vector<double> &initial, double sigma,
                   int max_iterations, int rand_seed = 0);

    std::shared_ptr<Solution> Optimize();

private:
    Cmaes cma_;

    double (*function_)(const std::vector<double> &);
    const std::vector<double> &initial_;
    int rand_seed_;
    int max_iterations_;
};

#endif //GISAXSMODELINGFRAMEWORK_CMAES_OPTIMIZER_H
