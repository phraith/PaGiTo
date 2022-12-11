//
// Created by phili on 29.10.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_WEIGHTS_INFORMATION_H
#define GISAXSMODELINGFRAMEWORK_WEIGHTS_INFORMATION_H

#include <eigen3/Eigen/Core>

class WeightsInformation {
public:
    WeightsInformation(Eigen::VectorX<double> weights, double c1, double cmu, double mu_eff);
    [[nodiscard]] const Eigen::VectorX<double> &Weights() const;
    [[nodiscard]] double C1() const;
    [[nodiscard]] double Cmu() const;
    [[nodiscard]] double MuEff() const;


private:
    Eigen::VectorX<double> weights_;
    double c1_;
    double cmu_;
    double mu_eff_;
};

#endif //GISAXSMODELINGFRAMEWORK_WEIGHTS_INFORMATION_H
