#ifndef GISAXS_LIB_GPU_FF_TRAPEZOID_CUH
#define GISAXS_LIB_GPU_FF_TRAPEZOID_CUH

#include "standard_vector_types.h"
#include "common/standard_defs.h"

class TrapezoidFF : public ShapeFF
{
public:
	__device__ TrapezoidFF(MyType2 beta, MyType2 L, MyType2 h, int rand_count);
	__device__ ~TrapezoidFF();
	__device__ MyComplex Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx);
	__device__ MyComplex Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx);
	__device__ MyComplex FF(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx);
	__device__ ShapeTypeV2 Type();
	__device__ int ParamCount();
	__device__ int BetaCount();
	__device__ MyType* RandBetas();
	__device__ MyType* RandLs();
	__device__ MyType* RandHs();
	__device__ MyType2* Beta();
	__device__ MyType2 L();
	__device__ MyType2 H();
	__device__ void Update(MyType2 new_L, MyType2 new_h);
	__device__ void UpdateBeta(MyType2 new_beta, int i);

private:
	MyType2 beta_[7];
	MyType2 L_;
	MyType2 h_;
	int rand_count_;
	MyType* rand_betas_;
	MyType* rand_Ls_;
	MyType* rand_hs_;
	ShapeTypeV2 type_;
};

#endif