#include "gpu/core/gpu_qgrid.h"
#include <cuComplex.h>

__device__ void real_space_to_q(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance, MyType2I real_pos,
                                MyType2I beam_pos, int q_idx, int qcount, GpuQGrid::GpuQGridContainer qgrid_container) {
    int pixel_x = real_pos.x + 1;
    int pixel_y = real_pos.y + 1;

    MyType pixel_dist_x = pixelsize * (pixel_x - beam_pos.x);
    MyType pixel_dist_y = pixelsize * (pixel_y - beam_pos.y);

    MyType theta_f = atan2f(pixel_dist_x, sample_distance);
    MyType quad_dist_x = sqrtf(sample_distance * sample_distance + pixel_dist_x * pixel_dist_x);
    MyType alpha_f = atan2f(pixel_dist_y, quad_dist_x);

    MyType qx = k0 * (cosf(alpha_f) * cosf(theta_f) - cosf(alpha_i));
    MyType qy = k0 * cosf(alpha_f) * sinf(theta_f);
    MyType qz = k0 * (sinf(alpha_f) + sinf(alpha_i));

    MyType qx2 = qx * qx;
    MyType qy2 = qy * qy;

    MyType qpar = sqrtf(qx2 + qy2);

    qgrid_container.dev_qpar[q_idx] = MyComplex{qpar, 0};
    qgrid_container.dev_qx[q_idx] = qx;
    qgrid_container.dev_qy[q_idx] = qy;
    qgrid_container.dev_qz[q_idx] = qz;

    MyType prefactor1 = sinf(alpha_f) * sinf(alpha_f);
    MyType prefactor2 = sinf(alpha_i) * sinf(alpha_i);
    MyComplex kz_af = cuCmulf(MyComplex{k0, 0}, cuCsqrt(cuCaddf(MyComplex{prefactor1, 0}, MyComplex{0, 0})));
    MyComplex kz_ai = cuCmulf(MyComplex{-k0, 0}, cuCsqrt(cuCaddf(MyComplex{prefactor2, 0}, MyComplex{0, 0})));

    qgrid_container.dev_alpha_fs[q_idx] = alpha_f;
    qgrid_container.dev_theta_fs[q_idx] = theta_f;

    qgrid_container.dev_qpoints_xy[q_idx] = MyComplex{qx, 0};
    qgrid_container.dev_qpoints_xy[qcount + q_idx] = MyComplex{qy, 0};

    MyComplex qz0 = cuCsubf(kz_af, kz_ai);
    MyComplex qz1 = cuCsubf(cuCmulf(kz_af, MyComplex{-1, 0}), kz_ai);
    MyComplex qz2 = cuCaddf(kz_af, kz_ai);
    MyComplex qz3 = cuCaddf(cuCmulf(kz_af, MyComplex{-1, 0}), kz_ai);

    qgrid_container.dev_qpoints_z_coeffs[q_idx] = qz0;
    qgrid_container.dev_qpoints_z_coeffs[q_idx + qcount] = qz1;
    qgrid_container.dev_qpoints_z_coeffs[q_idx + 2 * qcount] = qz2;
    qgrid_container.dev_qpoints_z_coeffs[q_idx + 3 * qcount] = qz3;

    MyComplex qz02 = cuCmulf(qz0, qz0);
    MyComplex qz12 = cuCmulf(qz1, qz1);
    MyComplex qz22 = cuCmulf(qz2, qz2);
    MyComplex qz32 = cuCmulf(qz3, qz3);

    MyComplex prefactor3 = MyComplex{qx2 + qy2, 0};
    MyComplex q1 = cuCsqrt(cuCaddf(prefactor3, qz02));
    MyComplex q2 = cuCsqrt(cuCaddf(prefactor3, qz12));
    MyComplex q3 = cuCsqrt(cuCaddf(prefactor3, qz22));
    MyComplex q4 = cuCsqrt(cuCaddf(prefactor3, qz32));
}

__global__ void
create_qgrid(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance, MyType2I *real_positions,
             MyType2I beam_pos, int detector_width, int qcount, GpuQGrid::GpuQGridContainer qgrid_container) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < qcount; i += blockDim.x * gridDim.x) { ;
        MyType2I real_pos = real_positions[i];
        real_space_to_q(alpha_i, k0, pixelsize, sample_distance, real_pos, beam_pos,
                        real_pos.y * detector_width + real_pos.x,
                        qcount, qgrid_container);
    }
}

__global__ void
create_qgrid_full_detector(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance,
                           MyType2I beam_pos, int detector_width, int detector_height,
                           GpuQGrid::GpuQGridContainer qgrid_container) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    int qcount = detector_height * detector_width;
    for (int i = tid; i < qcount; i += blockDim.x * gridDim.x) {
        int y = i / detector_width;
        int x = i % detector_width;

        real_space_to_q(alpha_i, k0, pixelsize, sample_distance, MyType2I{x, y}, beam_pos,
                        y * detector_width + x,
                        qcount, qgrid_container);
    }
}


void
GpuQGrid::CreateQGrid(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance, MyType2I *real_positions,
                      MyType2I beam_pos, int detector_width, int qcount, GpuQGrid::GpuQGridContainer qgrid_container,
                      cudaStream_t work_stream) {

    int threads = 128;
    int blocks = qcount / threads + 1;

    create_qgrid<<<blocks, threads, 0, work_stream>>>(alpha_i, k0, pixelsize, sample_distance, real_positions, beam_pos,
                                                      detector_width, qcount, qgrid_container);
}


void
GpuQGrid::CreateQGridFull(MyType alpha_i, MyType k0, MyType pixelsize, MyType sample_distance,
                          MyType2I beam_pos, int detector_width, int detector_height,
                          GpuQGrid::GpuQGridContainer qgrid_container,
                          cudaStream_t work_stream) {

    int threads = 128;
    int blocks = (detector_width * detector_height) / threads + 1;

    create_qgrid_full_detector<<<blocks, threads, 0, work_stream>>>(alpha_i, k0, pixelsize, sample_distance, beam_pos,
                                                                    detector_width, detector_height, qgrid_container);
}