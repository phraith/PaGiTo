//
// Created by Phil on 09.01.2022.
//

#include "parameter_definitions/substrate_setup.h"

SubstrateSetup::SubstrateSetup(MyType substrate_delta, MyType substrate_beta)
        : substrate_delta_(substrate_delta), substrate_beta_(substrate_beta) {

}

bool SubstrateSetup::operator==(const SubstrateSetup &substrate_setup) {
    return substrate_delta_ == substrate_setup.substrate_delta_ && substrate_beta_ == substrate_setup.substrate_beta_;
}

MyType SubstrateSetup::SubstrateDelta() const {
    return substrate_delta_;
}

MyType SubstrateSetup::SubstrateBeta() const {
    return substrate_beta_;
}
