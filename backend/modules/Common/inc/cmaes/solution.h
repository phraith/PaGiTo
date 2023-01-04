//
// Created by phili on 23.10.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SOLUTION_H
#define GISAXSMODELINGFRAMEWORK_SOLUTION_H

#include <Eigen/Core>

class Solution {
public:
    Solution(Eigen::VectorX<double> parameters, double fitness);

    const Eigen::VectorX<double> &Parameters() const;

    double Fitness() const;

    bool operator<(const Solution &s) const;

private:
    Eigen::VectorX<double> parameters_;
    double fitness_;
};

#endif //GISAXSMODELINGFRAMEWORK_SOLUTION_H
