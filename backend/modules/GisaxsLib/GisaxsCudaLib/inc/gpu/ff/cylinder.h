#ifndef GISAXS_LIB_GPU_FF_CYLINDER_CUH
#define GISAXS_LIB_GPU_FF_CYLINDER_CUH

#include "standard_vector_types.h"
#include "common/standard_defs.h"

class CylinderFF : public ShapeFF
{
public:
	__device__ CylinderFF(MyType2 radius, MyType2 height, int rand_count);
	__device__ ~CylinderFF();

	__device__ MyComplex Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx);
	__device__ MyComplex Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx);

	__device__ ShapeTypeV2 Type();
	__device__ int ParamCount();
	__device__ MyType* RandRads();
	__device__ MyType* RandHeights();

	__device__ MyType2 Radius();
	__device__ MyType2 Height();

	__device__ void Update(MyType2 new_radius, MyType2 new_height);

private:
	MyType2 radius_;
	MyType2 height_;

	int rand_count_;
	MyType* rand_rads_;
	MyType* rand_heights_;
	ShapeTypeV2 type_;
};

#endif