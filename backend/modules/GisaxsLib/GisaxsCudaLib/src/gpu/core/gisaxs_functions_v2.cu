#include "gpu/core/gisaxs_functions_v2.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include "common/standard_constants.h"
#include "gpu/util/cuda_numerics.h"
#include "common/flat_unitcell.h"
#include "gpu/core/gpu_helper.h"

namespace GisaxsV2 {
    typedef MyComplex(*formfactor_t)(MyComplex qpar, MyComplex q, MyComplex qz,
                                     int first_parameter_index, int first_random_index,
                                     MyType2 *parameters, MyType *randoms);

    __global__ void
    cuda_calculate_propagation_coefficients_full(int qcount, MyType2I resolution, MyType2I beam_direction, MyType pixelsize,
                                            MyType sample_distance, MyType k0, MyComplex sub_n2m1, MyType alpha_i,
                                            MyComplex *coefficients) {

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= qcount)
            return;

        MyType sin_ai = sinf(alpha_i);
        MyType kzi = -1.f * k0 * sin_ai;
        MyComplex tmp = cuCsqrt(sin_ai * sin_ai - sub_n2m1);
        MyComplex rki = (sin_ai - tmp) / (sin_ai + tmp);

        MyType quad_dist_x = std::sqrt(
                sample_distance * sample_distance + pixelsize * pixelsize);

        for (int i = tid; i < qcount; i += blockDim.x * gridDim.x) {
            int y = (i / resolution.x) + 1;

            MyType pixel_dist_y = pixelsize * (y - beam_direction.y);

            const auto alpha_f = atan2f(pixel_dist_y, quad_dist_x);

            MyType qz = k0 * (sinf(alpha_f) + sinf(alpha_i));
            MyType kzf = qz + kzi;

            if (kzf < 0) {
                coefficients[i] = {0, 0};
                coefficients[qcount + i] = {0, 0};
                coefficients[2 * qcount + i] = {0, 0};
                coefficients[3 * qcount + i] = {0, 0};
            } else {
                MyType sin_af = kzf / k0;
                tmp = cuCsqrt(sin_af * sin_af - sub_n2m1);
                MyComplex rkf = (sin_af - tmp) / (sin_af + tmp);

                MyComplex t4 = rki * rkf;

                coefficients[i] = {1, 0};
                coefficients[qcount + i] = rkf;
                coefficients[2 * qcount + i] = rki;
                coefficients[3 * qcount + i] = t4;
            }
        }
    }

    __global__ void
    cuda_calculate_propagation_coefficients(int qcount, MyType2I *detector_positions, MyType2I resolution, MyType2I beam_direction, MyType pixelsize,
                                                 MyType sample_distance, MyType k0, MyComplex sub_n2m1, MyType alpha_i,
                                                 MyComplex *coefficients) {

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= qcount)
            return;

        MyType sin_ai = sinf(alpha_i);
        MyType kzi = -1.f * k0 * sin_ai;
        MyComplex tmp = cuCsqrt(sin_ai * sin_ai - sub_n2m1);
        MyComplex rki = (sin_ai - tmp) / (sin_ai + tmp);

        MyType quad_dist_x = std::sqrt(
                sample_distance * sample_distance + pixelsize * pixelsize);

        for (int i = tid; i < qcount; i += blockDim.x * gridDim.x) {
            int y = detector_positions[i].y + 1;

            MyType pixel_dist_y = pixelsize * (y - beam_direction.y);

            const auto alpha_f = atan2f(pixel_dist_y, quad_dist_x);

            MyType qz = k0 * (sinf(alpha_f) + sinf(alpha_i));
            MyType kzf = qz + kzi;

            if (kzf < 0) {
                coefficients[i] = {0, 0};
                coefficients[qcount + i] = {0, 0};
                coefficients[2 * qcount + i] = {0, 0};
                coefficients[3 * qcount + i] = {0, 0};
            } else {
                MyType sin_af = kzf / k0;
                tmp = cuCsqrt(sin_af * sin_af - sub_n2m1);
                MyComplex rkf = (sin_af - tmp) / (sin_af + tmp);

                MyComplex t4 = rki * rkf;

                coefficients[i] = {1, 0};
                coefficients[qcount + i] = rkf;
                coefficients[2 * qcount + i] = rki;
                coefficients[3 * qcount + i] = t4;
            }
        }
    }

    __device__ MyComplex CalculateSphereFF(MyComplex qpar, MyComplex q, MyComplex qz,
                                           int first_parameter_index, int first_random_index,
                                           MyType2 *parameters, MyType *randoms) {
        MyType2 radius_base = parameters[first_parameter_index];
        MyType random_number = randoms[first_random_index];
        MyType radius = (random_number * radius_base.y) + radius_base.x;

        MyComplex qR = q * radius;

        if (cuC_abs(qR) < CUTINY_) {
            return {0, 0};
        }

        MyComplex sincos = cuCsin(qR) - cuCcos(qR) * qR;
        MyComplex expval = cuCexpi(qz * radius);

        MyComplex ff = (FOUR_PI_ / (q * q * q)) * expval * sincos;
        return ff;
    }

    __device__ MyComplex CalculateCylinderFF(MyComplex qpar, MyComplex q, MyComplex qz,
                                             int first_parameter_index, int first_random_index,
                                             MyType2 *parameters, MyType *randoms) {
        MyType2 radius_base = parameters[first_parameter_index];
        MyType radius_number = randoms[first_random_index];
        MyType radius = (radius_number * radius_base.y) + radius_base.x;

        MyType2 height_base = parameters[first_parameter_index + 1];
        MyType height_number = randoms[first_random_index + 1];
        MyType height = (height_number * height_base.y) + height_base.x;

        MyType temp1 = 2 * PI_ * radius * radius * height;
        MyComplex temp2 = {0, 0};

        MyComplex qr = qpar * radius;

        if ((qpar.x * qpar.x + qpar.y * qpar.y) > CUTINY_)
            temp2 = make_cuFloatComplex(j1f(cu_real(qr)), 0.0) / (qr);

        MyComplex temp3 = cuCsinc(0.5 * qz * height);
        MyComplex temp4 = cuCexpi(0.5 * qz * height);
        return temp1 * temp2 * temp3 * temp4;
    }

    __device__ __constant__  formfactor_t constNumber[2] =
            {
                    CalculateSphereFF,
                    CalculateCylinderFF
            };

    __device__ MyComplex EvalStructureFactor(MyComplex qx, MyComplex qy, MyComplex qz, float3 d, MyType n) {
        MyComplex r = qx * d.x + qy * d.y + qz * d.z;
        if (r.x == 0 && r.y == 0) { return {n, 0}; }
        return (1. - cuCexpi(-1.f * r * n)) / (1. - cuCexpi(-1.f * r));
    }

    __global__ void
    cuda_calc_sf(MyType3 distances, MyType3I repetitions, MyType2 *qxy, MyType2 *qz, MyComplex *sfs, int qcount) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= qcount) { return; }

        MyComplex qx = qxy[tid];
        MyComplex qy = qxy[qcount + tid];

        MyComplex qz1 = qz[tid];
        MyComplex qz2 = qz[qcount + tid];
        MyComplex qz3 = qz[2 * qcount + tid];
        MyComplex qz4 = qz[3 * qcount + tid];

        float3 dx = {distances.x, 0, 0};
        float3 dy = {0, distances.y, 0};
        float3 dz = {0, 0, distances.z};


        sfs[tid] = EvalStructureFactor(qx, qy, qz1, dx, repetitions.x)
                   * EvalStructureFactor(qx, qy, qz1, dy, repetitions.y)
                   * EvalStructureFactor(qx, qy, qz1, dz, repetitions.z);

        sfs[qcount + tid] = EvalStructureFactor(qx, qy, qz2, dx, repetitions.x)
                            * EvalStructureFactor(qx, qy, qz2, dy, repetitions.y)
                            * EvalStructureFactor(qx, qy, qz2, dz, repetitions.z);

        sfs[2 * qcount + tid] = EvalStructureFactor(qx, qy, qz3, dx, repetitions.x)
                                * EvalStructureFactor(qx, qy, qz3, dy, repetitions.y)
                                * EvalStructureFactor(qx, qy, qz3, dz, repetitions.z);

        sfs[3 * qcount + tid] = EvalStructureFactor(qx, qy, qz4, dx, repetitions.x)
                                * EvalStructureFactor(qx, qy, qz4, dy, repetitions.y)
                                * EvalStructureFactor(qx, qy, qz4, dz, repetitions.z);
    }


    __global__ void cuda_run_gisaxs_opt4(MyComplex *qpar, MyComplex *qabs, MyType2 *qxy, MyType2 *qz, int calculations,
                                         MyComplex *coefficients, MyType *intensities, int shape_count, int qcount,
                                         MyComplex *sfs, GisaxsV2::DeviceFlatUnitcell flat_unitcell, MyType *randoms) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= calculations)
            return;

        int idx = tid / COHERENCY_DRAW_RATIO.y;
        int loc_n = tid % COHERENCY_DRAW_RATIO.y;

        MyComplex qx = qxy[idx];
        MyComplex qy = qxy[qcount + idx];

        MyComplex qpar_idx = qpar[idx];
        MyComplex scattering = {0, 0};

        for (int k = 0; k < shape_count; ++k) {
            int loc_start_idx = flat_unitcell.position_indices[k];
            int loc_end_idx = flat_unitcell.position_indices[k + 1];
            int loc_count = loc_end_idx - loc_start_idx;

            auto shape_type = flat_unitcell.shapes[k];
            for (int i = 0; i < 4; ++i) {
                MyComplex qz_c = qz[i * qcount + idx];
                MyComplex qabs_c = qabs[i * qcount + idx];
                MyComplex sfs_c = sfs[i * qcount + idx];

                MyComplex shape_sum = {0, 0};
                for (int j = 0; j < COHERENCY_DRAW_RATIO.x; ++j) {
                    int current_iteration = j * COHERENCY_DRAW_RATIO.y + loc_n;
                    shape_sum = shape_sum +
                                constNumber[(int) shape_type](qpar_idx, qabs_c, qz_c,
                                                              flat_unitcell.parameter_indices[k],
                                                              current_iteration, flat_unitcell.parameters, randoms);


                }
                MyComplex shape_sum_u = {0, 0};
                for (int l = 0; l < loc_count; ++l) {
                    MyType3 loc = flat_unitcell.positions[loc_start_idx + l];
                    MyComplex qr = qx * loc.x + qy * loc.y + qz_c * loc.z;

                    shape_sum_u = shape_sum_u + shape_sum * cuCexpi(-1.f * qr);
                }
                scattering = scattering + coefficients[i * qcount + idx] * shape_sum_u * sfs_c;
            }
        }

        MyType scatter_abs = cuCabs(scattering);
        MyType intensity = scatter_abs * scatter_abs;

        if (!isnan(intensity))
            atomicAdd(&intensities[idx], intensity);
    }


    void RunSim(MyComplex *qpar, MyComplex *q, MyComplex *qpoints_xy, MyComplex *qpoints_z_coeffs, int qcount,
                MyComplex *coefficients, MyType *intensities, int shape_count, MyComplex *sfs,
                DeviceFlatUnitcell flat_unitcell,
                MyType *randoms, cudaStream_t work_stream) {
        int n = COHERENCY_DRAW_RATIO.y;

        int threads = 128;
        int calculations = qcount * n;
        int blocks = calculations / threads + 1;

        cuda_calc_sf << <
        qcount / threads +
        1, threads, 0, work_stream >> > (flat_unitcell.distances, flat_unitcell.repetitions, qpoints_xy, qpoints_z_coeffs, sfs, qcount);

        cuda_run_gisaxs_opt4 << <
        blocks, threads, 0, work_stream >> > (qpar, q, qpoints_xy, qpoints_z_coeffs, calculations, coefficients, intensities, shape_count, qcount, sfs, flat_unitcell, randoms);
    }

    void CalculatePropagationCoefficientsTopBuriedFull(int qcount, MyType2I resolution, MyType2I beam_direction,
                                                       MyType pixelsize,
                                                       MyType sample_distance, MyType k0, MyComplex sub_n2m1,
                                                       MyType alpha_i,
                                                       MyComplex *coefficients, cudaStream_t work_stream) {
        int threads = 128;
        int blocks = 128;

        cuda_calculate_propagation_coefficients_full<<< blocks, threads, 0, work_stream>>>(qcount, resolution,
                                                                                      beam_direction, pixelsize,
                                                                                      sample_distance, k0, sub_n2m1,
                                                                                      alpha_i, coefficients);
    }

    void CalculatePropagationCoefficientsTopBuried(int qcount, MyType2I *detector_positions, MyType2I resolution, MyType2I beam_direction,
                                                   MyType pixelsize,
                                                   MyType sample_distance, MyType k0, MyComplex sub_n2m1,
                                                   MyType alpha_i,
                                                   MyComplex *coefficients, cudaStream_t work_stream) {
        int threads = 128;
        int blocks = 128;

        cuda_calculate_propagation_coefficients<<< blocks, threads, 0, work_stream>>>(qcount, detector_positions, resolution,
                                                                                           beam_direction, pixelsize,
                                                                                           sample_distance, k0, sub_n2m1,
                                                                                           alpha_i, coefficients);
    }
}