#ifndef MODEL_FITTER_CORE_LAYER_H
#define MODEL_FITTER_CORE_LAYER_H

#include <complex>

#include "standard_vector_types.h"
#include "standard_defs.h"

class Layer
{
public:
	Layer(MyType delta, MyType beta, int order, MyType thickness);
	[[nodiscard]] const std::complex<MyType>& N2MinusOne() const;
	[[nodiscard]] MyType Z() const;
	void SetZ(MyType new_z);
	[[nodiscard]] MyType Thickness() const;

private:
	int order_;
	MyType thickness_;
	MyType zval_;
	std::complex<MyType> n2_minus_one_;
};

#endif