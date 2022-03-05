#ifndef MODEL_FITTER_CORE_SAMPLE_H
#define MODEL_FITTER_CORE_SAMPLE_H

#include <queue>
#include "vector_types.h"

#include "common/layer.h"
#include "standard_vector_types.h"
#include "parameter_definitions/detector_setup.h"
#include "common/beam_configuration.h"

struct TransRef
{
	std::complex<MyType> T;
	std::complex<MyType> R;
};

class SampleConfiguration
{
public:
	SampleConfiguration(Layer substrate, const std::vector<Layer>& layers);
	SampleConfiguration(const SampleConfiguration &sample);

	void InsertLayer(MyType delta, MyType beta, int order, MyType thickness);
    [[nodiscard]] const std::deque<Layer> &Layers() const;
	std::complex<MyType> TopMostN2();
	std::complex<MyType> N2M1OfLevel(int level);
	std::complex<MyType> SubstrateN2M1();
private:
	std::deque<Layer> layers_;
};

#endif