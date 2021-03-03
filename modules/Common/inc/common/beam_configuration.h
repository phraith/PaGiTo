#ifndef MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H
#define MODEL_SIMULATOR_CORE_BEAM_CONFIGURATION_H

#include "standard_vector_types.h"

#include <string>

class BeamConfiguration
{
public:
	BeamConfiguration(double alpha_i, MyType2 beam_direction, double wavelength, double coherency_ratio);

	double AlphaI() const;
	const MyType2& BeamDirection() const;
	double K0() const;

	std::string InfoStr() const;
private:
	double alpha_i_;
	MyType2 beam_direction_;
	double wavelength_;
	double coherency_ratio_;
	double k0_;
};

#endif