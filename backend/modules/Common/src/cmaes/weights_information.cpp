//
// Created by phili on 29.10.2022.
//

#include "cmaes/weights_information.h"
#include <utility>

WeightsInformation::WeightsInformation(Eigen::VectorX<double> weights, double c1, double cmu, double mu_eff)
        :
        weights_(std::move(weights)),
        c1_(c1),
        cmu_(cmu),
        mu_eff_(mu_eff) {

}

const Eigen::VectorX<double> &WeightsInformation::Weights() const {
    return weights_;
}

double WeightsInformation::C1() const {
    return c1_;
}

double WeightsInformation::Cmu() const {
    return cmu_;
}

double WeightsInformation::MuEff() const {
    return mu_eff_;
}
