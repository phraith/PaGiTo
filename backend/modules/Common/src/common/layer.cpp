#include "common/layer.h"

#include "common/standard_constants.h"

Layer::Layer(MyType delta, MyType beta, int order, MyType thickness)
	:
	order_(order),
	thickness_(thickness),
	n2_minus_one_(std::complex(MYTYPE_2 * delta, MYTYPE_2 * beta)),
	zval_(0)
{
}

const std::complex<MyType>& Layer::N2MinusOne() const
{
	return n2_minus_one_;
}

MyType Layer::Z() const
{
	return zval_;
}

void Layer::SetZ(MyType new_z)
{
	zval_ = new_z;
}

MyType Layer::Thickness() const
{
	return thickness_;
}
