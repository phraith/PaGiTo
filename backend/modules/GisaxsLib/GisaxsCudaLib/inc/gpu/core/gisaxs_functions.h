#ifndef GISAXS_LIB_GPU_GISAXS_FUNCTIONS_CUH
#define GISAXS_LIB_GPU_GISAXS_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include "gpu/util/cuda_numerics.h"
#include "common/standard_defs.h"
#include "gpu/ff/shape.h"

class DevUnitcell
{
public:
	__device__ DevUnitcell(ShapeTypeV2 *shape_types, int shape_count, int rand_count, MyType3* shape_locations, int *locations_counts, MyType3I repetitions, MyType3 distances);
	__device__ ~DevUnitcell();
	__device__ ShapeFF* GetShape(int idx);
	__device__ MyType3* ShapeLocations();
	__device__ int* LocationCounts();
	__device__ MyType3I Repetitions();
	__device__ MyType3 Distances();

private:
	ShapeFF** shapes_;
	int shape_count_;

	MyType3* shape_locations_;
	int* locations_counts_;

	MyType3I repetitions_;
	MyType3 distances_;
};

namespace Gisaxs
{
	void CreateUnitcell(DevUnitcell** dev_unitcell, ShapeTypeV2* shape_types, int shape_count, MyType3* locations, int *locations_counts, int rand_count, MyType3I repetitions, MyType3 distances, cudaStream_t work_stream);
	void DestroyUnitcell(cudaStream_t work_stream);
	void RunSim(MyComplex* qpar, MyComplex* q, MyComplex* qpoints_xy, MyComplex* qpoints_z_coeffs, int qcount, MyComplex* coefficients, MyType* intensities, int shape_count, MyComplex *sfs, cudaStream_t work_stream);
	void Update(MyType* rands, int qcount, MyType2* params, int shape_count, cudaStream_t work_stream);
}
#endif