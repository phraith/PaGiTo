//
// Created by Phil on 09.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SUBSTRATE_SETUP_H
#define GISAXSMODELINGFRAMEWORK_SUBSTRATE_SETUP_H


#include "standard_vector_types.h"

class SubstrateSetup {
public:
    SubstrateSetup(MyType substrate_delta, MyType substrate_beta);

    bool operator==(const SubstrateSetup &substrate_setup);

private:
    MyType substrate_delta_;
    MyType substrate_beta_;
public:
    MyType SubstrateDelta() const;

    MyType SubstrateBeta() const;
};


#endif //GISAXSMODELINGFRAMEWORK_SUBSTRATE_SETUP_H
