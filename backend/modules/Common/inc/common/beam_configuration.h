#ifndef MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H
#define MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H

#include "standard_vector_types.h"

#include <string>

class BeamConfiguration
{
public:
	BeamConfiguration(double deg_alpha_i, MyType2I beam_direction, double photon_ev, double coherency_ratio);

	[[nodiscard]] double AlphaI() const;
	[[nodiscard]] const MyType2I& BeamDirection() const;
	[[nodiscard]] double K0() const;

	std::string InfoStr() const;
private:
	double alpha_i_;
	MyType2I beam_direction_;
	double wavelength_;
	double coherency_ratio_;
	double k0_;
};

#endif