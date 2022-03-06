#ifndef GISAXS_LIB_GPU_FF_SPHERE_CUH
#define GISAXS_LIB_GPU_FF_SPHERE_CUH

#include "common/standard_defs.h"
#include "shape.h"
#include "gpu/util/util.h"

class SphereFF : public ShapeFF
{
public:
	__device__ SphereFF(MyType2 radius, int rand_count);
	__device__ ~SphereFF();
	__device__ MyComplex Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx);
	__device__ MyComplex Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx);
	__device__ ShapeTypeV2 Type();
	__device__ int ParamCount();
	__device__ MyType* RandRads();
	__device__ MyType2 Radius();
	__device__ void Update(MyType2 new_radius);

private:
	MyType2 radius_;
	int rand_count_;
	MyType* rand_rad_;
	ShapeTypeV2 type_;
};

#endif