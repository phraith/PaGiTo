#ifndef MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H
#define MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H

#include "standard_defs.h"

#include <string>

class BeamConfiguration
{
public:
	BeamConfiguration(double deg_alpha_i, Vector2<int> beam_direction, double photon_ev, double coherency_ratio);

	[[nodiscard]] double AlphaI() const;
	[[nodiscard]] const Vector2<int>& BeamDirection() const;
	[[nodiscard]] double K0() const;

	[[nodiscard]] std::string InfoStr() const;
private:
	double alpha_i_;
    Vector2<int> beam_direction_;
	double wavelength_;
	double coherency_ratio_;
	double k0_;
};

#endif