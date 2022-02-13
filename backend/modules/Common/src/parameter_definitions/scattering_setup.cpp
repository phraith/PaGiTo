//
// Created by Phil on 09.01.2022.
//

#include "parameter_definitions/scattering_setup.h"

ScatteringSetup::ScatteringSetup(MyType alphai, MyType ev) : alphai_(alphai), ev_(ev) {}

bool ScatteringSetup::operator==(const ScatteringSetup &scattering_setup) {
    return alphai_ == scattering_setup.alphai_ && ev_ == scattering_setup.ev_;
}

MyType ScatteringSetup::Ev() const {
    return ev_;
}

MyType ScatteringSetup::Alphai() const {
    return alphai_;
}
