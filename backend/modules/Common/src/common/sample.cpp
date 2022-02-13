#include "common/sample.h"

#include <algorithm>

#include "common/standard_constants.h"
#include <iostream>
#include <cassert>

Sample::Sample(Layer substrate, std::vector<Layer> layers)
	:
	layers_({ Layer {0, 0, 0 , 0} })
{
	for (const auto &layer : layers)
	{
		layers_.emplace_back(layer);
	}
	layers_.emplace_back(substrate);

	MyType z = 0;
	for (auto &layer : layers_)
	{
		z -= layer.Thickness();

		layer.SetZ(z);
	}
}

Sample::Sample(const Sample &sample)
{
	this->layers_ = sample.layers_;
}

void Sample::InsertLayer(MyType delta, MyType beta, int order, MyType thickness)
{
	layers_.emplace_front(Layer(delta, beta, order, thickness));
}

TransRef Sample::ParratsRecursion(MyType alpha, MyType k0, int order) const
{
	std::vector<std::complex<MyType>> T(layers_.size(), 0);
	std::vector<std::complex<MyType>> R(layers_.size(), 0);
	std::vector<std::complex<MyType>> k(layers_.size(), 0);

	MyType sina = std::sin(alpha);

	//calculate k values
	std::transform(layers_.begin(), layers_.end(), k.begin(),
		[&](Layer l) -> std::complex<MyType> { return -k0 * std::sqrt(sina * sina - l.N2MinusOne()); });

	//init T_n with 1
	T[0] = std::complex<MyType>(1, 0);

	std::complex<MyType> j(0, 1);

	for (int i = layers_.size() - 2; i > -1; --i)
	{
		std::complex<MyType> rii1 = (k[i] - k[i + 1]) / (k[i] + k[i + 1]);
		std::complex<MyType> xi1 = 0;
		if (R[i + 1] != 0.f)
		 xi1 = R[i + 1] / T[i + 1];

		MyType z = layers_.at(i).Thickness();
		std::complex<MyType> pexp = std::exp(MYTYPE_2 * j * k[i+1] * z);
		std::complex<MyType> mexp = std::exp(-MYTYPE_2 * j * k[i] * z);

		R[i] = (rii1 + xi1 * pexp) * mexp;
		T[i] = (MYTYPE_1 + rii1 * xi1 * pexp);

	}
	
	

	//normalize
	std::complex t0(T[0]);
	for (int i = 0; i < layers_.size(); ++i)
	{
		T[i] /= t0;
		R[i] /= t0;
	}

	return TransRef{ T[order], R[order] };
}

std::vector<MyComplex4> Sample::PropagationCoeffs(MyType alpha_i, const std::vector<MyType> &alpha_fs, MyType k0, MyType order) const
{
	TransRef TRi(ParratsRecursion(alpha_i, k0, order));
	std::vector<TransRef> TRfs(alpha_fs.size());

	//calculate TRfs values
	std::transform(alpha_fs.begin(), alpha_fs.end(), TRfs.begin(),
		[&](MyType alpha_f) -> TransRef { return (alpha_f > 0) ? ParratsRecursion(alpha_f, k0, order) : TransRef{0,0}; });

	std::vector<MyComplex4> trans_refs;
	for (const auto & TRf : TRfs)
	{
		std::complex<MyType> t1 = TRi.T * TRf.T;
		std::complex<MyType> t2 = TRi.T * TRf.R;
		std::complex<MyType> t3 = TRi.R * TRf.T;
		std::complex<MyType> t4 = TRi.R * TRf.R;

		trans_refs.emplace_back(MyComplex4{ {t1.real(), t1.imag()}, {t2.real(), t2.imag()}, {t3.real(), t3.imag()}, {t4.real(), t4.imag()} });
	}
	return trans_refs;
}

std::vector<MyComplex> Sample::PropagationCoeffsTopBuried(MyType alpha_i, const std::vector<MyType>& alpha_fs, MyType k0) const
{
	std::vector<MyComplex> trans_refs(4 * alpha_fs.size());
	auto& substrate = layers_.at(layers_.size() - 1);
	const auto &ns2m1 = substrate.N2MinusOne();

	MyType sin_ai = std::sin(alpha_i);
	MyType kzi = -1 * k0 * sin_ai;
	std::complex<MyType> tmp = std::sqrt(sin_ai * sin_ai - ns2m1);
	std::complex<MyType> rki = (sin_ai - tmp) / (sin_ai + tmp);
	int qcount = alpha_fs.size();
	for (int i = 0; i < qcount; ++i)
	{
		const auto& alpha_f = alpha_fs[i];

		MyType qz = k0 * (std::sin(alpha_f) + std::sin(alpha_i));
		MyType kzf = qz + kzi;

		if (kzf < 0)
		{
			trans_refs[i] = {0,0};
			trans_refs[qcount + i] = { 0,0 };
			trans_refs[2 * qcount + i] = { 0,0 };
			trans_refs[3 * qcount + i] = { 0,0 };
		}
		else
		{
			MyType sin_af = kzf / k0;
			tmp = std::sqrt(sin_af * sin_af - ns2m1);
			std::complex<MyType> rkf = (sin_af - tmp) / (sin_af + tmp);

			std::complex<MyType> t4 = rki * rkf;

			trans_refs[i] = { 1,0 };
			trans_refs[qcount + i] = { rkf.real(), rkf.imag() };
			trans_refs[2 * qcount + i] = { rki.real(), rki.imag() };
			trans_refs[3 * qcount + i] = { t4.real(), t4.imag() };

			//trans_refs.emplace_back(MyComplex4{ {1,0}, {rkf.real(), rkf.imag()}, {rki.real(), rki.imag()}, {t4.real(), t4.imag()} });
		}
	}

	return trans_refs;
}

const std::complex<MyType> Sample::TopMostN2()
{
	return layers_.at(0).N2MinusOne();
}

const std::complex<MyType> Sample::N2M1OfLevel(int level)
{
	if (level >= layers_.size())
		return { 0,0 };

	return layers_.at(level).N2MinusOne();
}

const std::complex<MyType> Sample::SubstrateN2M1()
{
	assert(layers_.size() > 0);

	return layers_.at(layers_.size() - 1).N2MinusOne();
}