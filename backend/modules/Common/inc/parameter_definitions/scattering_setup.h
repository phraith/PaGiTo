//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SCATTERING_SETUP_H
#define GISAXSMODELINGFRAMEWORK_SCATTERING_SETUP_H

#include "standard_vector_types.h"

class ScatteringSetup {
public:

    ScatteringSetup(MyType alphai, MyType ev);

    bool operator==(const ScatteringSetup &scattering_setup);

private:
    MyType alphai_;
public:
    MyType Alphai() const;

private:
    MyType ev_;
public:
    MyType Ev() const;
};


#endif //GISAXSMODELINGFRAMEWORK_SCATTERING_SETUP_H
