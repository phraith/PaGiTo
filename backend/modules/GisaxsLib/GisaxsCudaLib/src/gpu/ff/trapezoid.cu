#include "gpu/ff/trapezoid.h"

#include "gpu/util/cuda_numerics.h"

__device__ TrapezoidFF::TrapezoidFF(MyType2 beta, MyType2 L, MyType2 h, int rand_count)
	:
	L_(L),
	h_(h),
	rand_count_(rand_count),
	rand_betas_(new MyType[BetaCount() * rand_count]),
	rand_Ls_(new MyType[rand_count_]),
	rand_hs_(new MyType[rand_count_]),
	type_(ShapeTypeV2::cylinder)
{
}

__device__ TrapezoidFF::~TrapezoidFF()
{
	if (rand_betas_ != nullptr)
		delete[] rand_betas_;

	if (rand_Ls_ != nullptr)
		delete[] rand_Ls_;

	if (rand_hs_ != nullptr)
		delete[] rand_hs_;
}

__device__ MyComplex TrapezoidFF::FF(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx)
{
	MyType h = rand_hs_[rand_idx];
	MyType L = rand_Ls_[rand_idx];

	MyType beta = rand_betas_[rand_idx] * PI_ / 180.f;

	MyType m1 = tanf(beta);
	MyType m2 = tanf(PI_ - beta);

	MyType y1 = -L / 2;
	MyType y2 = L / 2;

	MyComplex t1 = qy + m1 * qz;
	MyComplex t2 = qy + m2 * qz;

	MyComplex t3 = m1 * cuCexpi(-1 * qy * y1) * (1 - cuCexpi(-1 * h / m1 * t1)) / t1;
	MyComplex t4 = m2 * cuCexpi(-1 * qy * y2) * (1 - cuCexpi(-1 * h / m2 * t2)) / t2;

	return (t4 - t3) / qy;
}

__device__ MyComplex TrapezoidFF::Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx)
{
	MyType h = rand_hs_[rand_idx];
	MyType L = rand_Ls_[rand_idx];

	MyComplex ff = { 0,0 };

	for (int i = 0; i < BetaCount(); ++i)
	{
		MyType beta = rand_betas_[i * rand_count_ + rand_idx] * PI_ / 180.f;
		MyType shift = h * i;

		MyType m1 = tanf(beta);
		MyType m2 = tanf(PI_ - beta);

		MyType y1 = -L / 2;
		MyType y2 = L / 2;

		MyComplex t1 = qy + m1 * qz;
		MyComplex t2 = qy + m2 * qz;

		MyComplex t3 = m1 * cuCexpi(-1 * qy * y1) * (1 - cuCexpi(-1 * h / m1 * t1)) / t1;
		MyComplex t4 = m2 * cuCexpi(-1 * qy * y2) * (1 - cuCexpi(-1 * h / m2 * t2)) / t2;


		ff = ff + (t4 - t3) / qy * cuCexpi(-1 * shift * qz);

		L = (h / m2) - (h / m1);
	}

	return ff;
}

__device__ MyComplex TrapezoidFF::Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx)
{
	return { 0,0 };
}

__device__ ShapeTypeV2 TrapezoidFF::Type()
{
	return type_;
}

__device__ int TrapezoidFF::ParamCount()
{
	return 3;
}

__device__ int TrapezoidFF::BetaCount()
{
	return 1;
}

__device__ MyType* TrapezoidFF::RandBetas()
{
	return rand_betas_;
}

__device__ MyType* TrapezoidFF::RandLs()
{
	return rand_Ls_;
}

__device__ MyType* TrapezoidFF::RandHs()
{
	return rand_hs_;
}

__device__ MyType2 *TrapezoidFF::Beta()
{
	return beta_;
}

__device__ MyType2 TrapezoidFF::L()
{
	return L_;
}

__device__ MyType2 TrapezoidFF::H()
{
	return h_;
}

__device__ void TrapezoidFF::UpdateBeta(MyType2 new_beta, int i)
{
	beta_[i] = new_beta;
}

__device__ void TrapezoidFF::Update(MyType2 new_L, MyType2 new_h)
{
	L_ = new_L;
	h_ = new_h;
}
