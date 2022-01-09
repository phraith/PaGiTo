#include "common/beam_configuration.h"

#define _USE_MATH_DEFINES 
#include <math.h>



BeamConfiguration::BeamConfiguration(double alpha_i, MyType2 beam_direction, double wavelength, double coherency_ratio)
	:
	alpha_i_(alpha_i),
	beam_direction_(beam_direction),
	wavelength_(wavelength),
	coherency_ratio_(coherency_ratio),
	k0_(2 * M_PI / wavelength)
{
}

double BeamConfiguration::AlphaI() const
{
	return alpha_i_;
}

const MyType2& BeamConfiguration::BeamDirection() const
{
	return beam_direction_;
}

double BeamConfiguration::K0() const
{
	return k0_;
}

std::string BeamConfiguration::InfoStr() const
{
	std::string info = "Beam configuration:\n";
	info += "	-alpha_i (in rad): " + std::to_string(alpha_i_) + "\n";
	info += "	-wavelength (in nm): " + std::to_string(wavelength_) + "\n";
	info += "	-coherency ratio: " + std::to_string(coherency_ratio_) + "\n";
	info += "	-k0 (in 1/nm): " + std::to_string(k0_) + "\n";
	return info;
}
