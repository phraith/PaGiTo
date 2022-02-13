#ifndef MODEL_FITTER_CORE_SAMPLE_H
#define MODEL_FITTER_CORE_SAMPLE_H

#include <queue>
#include "vector_types.h"

#include "common/layer.h"
#include "standard_vector_types.h"
struct TransRef
{
	std::complex<MyType> T;
	std::complex<MyType> R;
};

class Sample
{
public:
	Sample(Layer substrate, std::vector<Layer> layers);
	Sample(const Sample &sample);

	void InsertLayer(MyType delta, MyType beta, int order, MyType thickness);
	TransRef ParratsRecursion(MyType alpha, MyType k0, int order) const;
	std::vector<MyComplex4> PropagationCoeffs(MyType alpha_i, const std::vector<MyType>& alpha_fs, MyType k0, MyType order) const;
	std::vector<MyComplex> PropagationCoeffsTopBuried(MyType alpha_i, const std::vector<MyType>& alpha_fs, MyType k0) const;

	const std::complex<MyType> TopMostN2();
	const std::complex<MyType> N2M1OfLevel(int level);
	const std::complex<MyType> SubstrateN2M1();
private:
	std::deque<Layer> layers_;
};

#endif