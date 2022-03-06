#include "common/beam_configuration.h"

#define _USE_MATH_DEFINES 
#include <cmath>



BeamConfiguration::BeamConfiguration(double deg_alpha_i, Vector2<int> beam_direction, double photon_ev, double coherency_ratio)
	:
	alpha_i_(deg_alpha_i * 0.017453),
	beam_direction_(beam_direction),
	wavelength_(1239.84 / photon_ev),
	coherency_ratio_(coherency_ratio),
	k0_(2 * M_PI / wavelength_)
{
}

double BeamConfiguration::AlphaI() const
{
	return alpha_i_;
}

const Vector2<int>& BeamConfiguration::BeamDirection() const
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
