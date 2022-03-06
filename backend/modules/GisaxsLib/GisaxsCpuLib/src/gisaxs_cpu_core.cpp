//
// Created by Phil on 26.02.2022.
//
#define _USE_MATH_DEFINES

#include <complex>
#include <map>
#include "../inc/gisaxs_cpu_core.h"
#include "common/standard_constants.h"

typedef std::complex<MyType> (*formfactor)(std::complex<MyType> qpar, std::complex<MyType> q, std::complex<MyType> qz,
                                           int first_parameter_index, int first_random_index,
                                           const std::vector<Vector2<MyType>> &parameters, const std::vector<MyType> &randoms);

const std::map<ShapeTypeV2, formfactor> shape_type_to_formfactor{
        {ShapeTypeV2::sphere, GisaxsCpuCore::CalculateSphereFF},
        {ShapeTypeV2::cylinder, GisaxsCpuCore::CalculateCylinderFF},
};


std::vector<std::complex<MyType>>
GisaxsCpuCore::CalculateStructureFactors(const std::vector<std::complex<MyType>> &qxy,
                                         const std::vector<std::complex<MyType>> &qz,
                                         Vector3<MyType> distances, Vector3<int> repetitions) {
    unsigned long qcount = qxy.size() / 2;
    std::vector<std::complex<MyType>> structure_factors(4 * qcount);
    for (
            int i = 0;
            i < qcount;
            ++i) {
        auto qx = qxy[i];
        auto qy = qxy[qcount + i];

        auto qz1 = qz[i];
        auto qz2 = qz[i + qcount];
        auto qz3 = qz[i + 2 * qcount];
        auto qz4 = qz[i + 3 * qcount];

        Vector3<MyType> dx = {distances.x, 0, 0};
        Vector3<MyType> dy = {0, distances.y, 0};
        Vector3<MyType> dz = {0, 0, distances.z};

        structure_factors[i] =
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dx, repetitions
                        .x)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dy, repetitions
                        .y)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dz, repetitions
                        .z);
        structure_factors[i + qcount] =
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dx, repetitions
                        .x)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dy, repetitions
                        .y)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dz, repetitions
                        .z);

        structure_factors[i + 2 * qcount] =
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dx, repetitions
                        .x)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dy, repetitions
                        .y)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dz, repetitions
                        .z);

        structure_factors[i + 3 * qcount] =
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dx, repetitions
                        .x)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dy, repetitions
                        .y)
                *
                GisaxsCpuCore::EvaluateStructureFactor(qx, qy, qz1, dz, repetitions
                        .z);
    }

    return
            structure_factors;
}

std::complex<MyType>
GisaxsCpuCore::EvaluateStructureFactor(const std::complex<MyType> &qx, const std::complex<MyType> &qy,
                                       const std::complex<MyType> &qz, Vector3<MyType> d, MyType n) {
    std::complex<MyType> r = qx * d.x + qy * d.y + qz * d.z;
    if (r.real() == 0 && r.imag() == 0) {
        return std::complex<MyType>{n, 0};
    }

    return 1.f - Eiz((r * n * -1.f)) / Eiz((r * -1.f));
}

std::complex<MyType>
GisaxsCpuCore::CalculateSphereFF(std::complex<MyType> qpar, std::complex<MyType> q, std::complex<MyType> qz,
                                 int first_parameter_index, int first_random_index,
                                 const std::vector<Vector2<MyType>> &parameters, const std::vector<MyType> &randoms) {

    Vector2<MyType> radius_base = parameters.at(first_parameter_index);
    MyType random_number = randoms.at(first_random_index);

    MyType radius = (random_number * radius_base.y) + radius_base.x;

    std::complex<MyType> qr = q * radius;
    if (abs(radius) < 1.0e-13) {
        return 0;
    }

    auto sincos = sin(qr) - cos(qr) * qr;
    auto expval = Eiz(qz * radius);
    auto ff = (static_cast<MyType>(4.0 * M_PI) / (q * q * q)) * expval * sincos;
    return ff;
}

std::complex<MyType>
GisaxsCpuCore::CalculateCylinderFF(std::complex<MyType> qpar, std::complex<MyType> q, std::complex<MyType> qz,
                                 int first_parameter_index, int first_random_index,
                                 const std::vector<Vector2<MyType>> &parameters, const std::vector<MyType> &randoms) {
    Vector2<MyType> radius_base = parameters.at(first_parameter_index);
    MyType random_number = randoms.at(first_random_index);
    MyType radius = (random_number * radius_base.y) + radius_base.x;

    Vector2<MyType> height_base = parameters.at(first_parameter_index + 1);
    MyType height_number = randoms.at(first_random_index + 1);

    MyType height = (height_number * height_base.y) + height_base.x;

    MyType     temp1 = 2.f * M_PI * radius * radius * height;
    std::complex<MyType> temp2 = { 0,0 };

    if ((qpar.real() * qpar.real() + qpar.imag() * qpar.imag()) > 1.0e-13)
    {
        auto qpar_radius = qpar * radius;
        temp2 = std::complex<MyType>(std::cyl_bessel_jf(1, qpar_radius.real()), 0.0) / qpar_radius;
    }

    std::complex<MyType> temp3 = Sinc(0.5f * qz * height);
    std::complex<MyType> temp4 = Eiz(0.5f * qz * height);

    std::complex<MyType> ff = temp1 * temp2 * temp3 * temp4;
    return ff;
}
std::complex<MyType> GisaxsCpuCore::Eiz(std::complex<MyType> z) {
    return std::exp(std::complex {-z.imag(), z.real()});
}

std::complex<MyType>  GisaxsCpuCore::Sinc(std::complex<MyType> z) {
    if(abs(z.real()) < 1e-9 && abs(z.imag()) < 1e-9) return { 1.0, 0.0 };
    else return std::sin(z) / z;
}

std::vector<MyType>
GisaxsCpuCore::CalculateIntensities(const std::vector<std::complex<MyType>> &qpar,
                                    const std::vector<std::complex<MyType>> &q,
                                    const std::vector<std::complex<MyType>> &qpoints_xy,
                                    const std::vector<std::complex<MyType>> &qpoints_z_coeffs,
                                    const std::vector<std::complex<MyType>> &coefficients,
                                    const FlatUnitcellV2 &flat_unitcell,
                                    const std::vector<MyType> &randoms,
                                    const std::vector<std::complex<MyType>> &sfs) {
    unsigned long qcount = sfs.size() / 4;
    std::vector<MyType> intensities(qcount);
    for (int i = 0; i < qcount; ++i) {
        MyType intensity = 0;
        for (int n = 0; n < COHERENCY_DRAW_RATIO.y; ++n) {

            auto qx = qpoints_xy[i];
            auto qy = qpoints_xy[i + qcount];

            auto qpar_idx = qpar[i];
            std::complex<MyType> scattering = 0;
            for (int j = 0; j < flat_unitcell.ShapeTypes().size(); ++j) {
                auto shape_type = flat_unitcell.ShapeTypes()[j];
                for (int k = 0; k < 4; ++k) {
                    auto qz_c = qpoints_z_coeffs[k * qcount + i];
                    auto qabs_c = q[k * qcount + i];
                    auto sfs_c = sfs[k * qcount + i];


                    int loc_start_idx = flat_unitcell.PositionIndices().at(j);
                    int loc_end_idx = (j + 1 < flat_unitcell.PositionIndices().size())
                                      ? flat_unitcell.PositionIndices().at(
                                    j + 1) : flat_unitcell.Positions().size();
                    int loc_count = loc_end_idx - loc_start_idx;

                    int first_parameter_idx = flat_unitcell.ParameterIndices().at(j);
                    std::complex<MyType> shape_sum = 0;
                    for (int l = 0; l < COHERENCY_DRAW_RATIO.x; ++l) {
                        int rand_idx = 2 * first_parameter_idx * n * COHERENCY_DRAW_RATIO.x + l;
                        shape_sum = shape_sum +
                                    shape_type_to_formfactor.at(shape_type)(qpar_idx, qabs_c, qz_c, first_parameter_idx,
                                                                            rand_idx, flat_unitcell.Parameters(),
                                                                            randoms);
                    }

                    std::complex<MyType> shape_sum_u = {0, 0};
                    for (int l = 0; l < loc_count; ++l) {
                        Vector3<MyType> loc = flat_unitcell.Positions()[loc_start_idx + l];
                        std::complex<MyType> qr = qx * loc.x + qy * loc.y + qz_c * loc.z;
                        shape_sum_u = shape_sum_u + shape_sum * exp(-1.f * qr);
                    }
                    auto coeff = coefficients[k * qcount + i];
                    scattering = scattering + std::complex<MyType>(coeff.real(), coeff.imag()) * shape_sum_u * sfs_c;
                }
            }
            MyType scatterAbs = abs(scattering);
            MyType local_intensity = scatterAbs * scatterAbs;

            if (!std::isnan(local_intensity)) {
                intensity += local_intensity;
            }
        }
        intensities[i] = intensity;
    }

    return intensities;
}
