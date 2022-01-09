#ifndef MODEL_SIMULATOR_CORE_DETECTOR_H
#define MODEL_SIMULATOR_CORE_DETECTOR_H

#include <string>

#include "standard_vector_types.h"

class Detector
{
public:
	Detector(double pixel_size, MyType2I resolution, MyType2 direct_beam_location);

	double PixelSize() const;
	const MyType2I&Resolution() const;
	const MyType2& DirectBeamLocation() const;

	std::string InfoStr() const;
private:
	double pixel_size_;
	MyType2I resolution_;
	MyType2 direct_beam_location_;
};

#endif