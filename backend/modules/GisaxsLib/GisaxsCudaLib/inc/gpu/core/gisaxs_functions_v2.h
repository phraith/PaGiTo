#ifndef GISAXS_LIB_GPU_GISAXS_FUNCTIONS2_CUH
#define GISAXS_LIB_GPU_GISAXS_FUNCTIONS2_CUH

#include "cuda_runtime.h"
#include "gpu/util/cuda_numerics.h"

#include "curand.h"
#include "curand_kernel.h"

#include "common/standard_defs.h"

namespace GisaxsV2 {

    struct DeviceFlatUnitcell {
        MyType2 *parameters;
        int *parameter_indices;
        MyType3 *positions;
        int *position_indices;
        ShapeTypeV2 *shapes;

        MyType3I repetitions;
        MyType3 distances;
    };

    void RunSim(MyComplex *qpar, MyComplex *q, MyComplex *qpoints_xy, MyComplex *qpoints_z_coeffs, int qcount,
                MyComplex *coefficients, MyType *intensities, int shape_count, MyComplex *sfs,
                DeviceFlatUnitcell flat_unitcell, MyType *randoms,cudaStream_t work_stream);

    void CalculatePropagationCoefficientsTopBuriedFull(int qcount, MyType2I resolution, MyType2I beam_direction,
                                                       MyType pixelsize,
                                                       MyType sample_distance, MyType k0, MyComplex sub_n2m1,
                                                       MyType alpha_i,
                                                       MyComplex *coefficients, cudaStream_t work_stream);

    void CalculatePropagationCoefficientsTopBuried(int qcount, MyType2I *detector_positions, MyType2I resolution, MyType2I beam_direction,
                                                   MyType pixelsize,
                                                   MyType sample_distance, MyType k0, MyComplex sub_n2m1,
                                                   MyType alpha_i,
                                                   MyComplex *coefficients, cudaStream_t work_stream);
}
#endif