#ifndef GISAXS_LIB_GPU_GPU_QGRID_CUH
#define GISAXS_LIB_GPU_GPU_QGRID_CUH

#include "cuda_runtime.h"
#include "gpu/util/cuda_numerics.h"

#include "curand.h"
#include "curand_kernel.h"

#include "common/standard_defs.h"
#include "standard_vector_types.h"


namespace GpuQGrid {
    struct GpuQGridContainer {
        MyComplex *dev_qpoints_xy;
        MyComplex *dev_qpoints_z_coeffs;
        MyComplex *dev_qpar;
        MyComplex *dev_q;
        MyComplex *dev_coefficients;
        MyType *dev_alpha_fs;
        MyType *dev_theta_fs;
        MyType *dev_qx;
        MyType *dev_qy;
        MyType *dev_qz;
    };

    void CreateQGrid(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance, MyType2I *real_positions,
                     MyType2I beam_pos, int detector_width, int qcount, GpuQGrid::GpuQGridContainer qgrid_container,
                     cudaStream_t work_stream);


    void CreateQGridFull(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance,
                         MyType2I beam_pos, int detector_width, int detector_height,
                         GpuQGrid::GpuQGridContainer qgrid_container,
                         cudaStream_t work_stream);
}
#endif