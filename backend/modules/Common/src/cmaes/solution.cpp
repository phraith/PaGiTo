//
// Created by phili on 23.10.2022.
//

#include "cmaes/solution.h"

#include <utility>

const Eigen::VectorX<double> &Solution::Parameters() const {
    return parameters_;
}

double Solution::Fitness() const {
    return fitness_;
}

bool Solution::operator<(const Solution &s) const {
    return (fitness_ < s.Fitness());
}

Solution::Solution(Eigen::VectorX<double> parameters, double fitness)
        :
        parameters_(std::move(parameters)),
        fitness_(fitness) {

}
