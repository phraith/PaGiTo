//
// Created by Phil on 26.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_GISAXSCPUCORE_H
#define GISAXSMODELINGFRAMEWORK_GISAXSCPUCORE_H


#include <vector>
#include <complex>
#include "common/flat_unitcell.h"

namespace GisaxsCpuCore {
    std::vector<std::complex<MyType>> CalculateStructureFactors(const std::vector<std::complex<MyType>> &qxy,
                                                                const std::vector<std::complex<MyType>> &qz,
                                                                Vector3<MyType> distances,
                                                                Vector3<int> repetitions);

    std::complex<MyType>
    EvaluateStructureFactor(const std::complex<MyType> &qx, const std::complex<MyType> &qy,
                            const std::complex<MyType> &qz, Vector3<MyType> d, MyType n);

    std::vector<MyType> CalculateIntensities(const std::vector<std::complex<MyType>> &qpar,
                                             const std::vector<std::complex<MyType>> &q,
                                             const std::vector<std::complex<MyType>> &qpoints_xy,
                                             const std::vector<std::complex<MyType>> &qpoints_z_coeffs,
                                             const std::vector<std::complex<MyType>> &coefficients,
                                             const FlatUnitcellV2 &flat_unitcell,
                                             const std::vector<MyType> &randoms,
                                             const std::vector<std::complex<MyType>> &sfs);

    std::complex<MyType>
    CalculateSphereFF(std::complex<MyType> qpar, std::complex<MyType> q, std::complex<MyType> qz,
                      int first_parameter_index, int first_random_index, const std::vector<Vector2<MyType>> &parameters,
                      const std::vector<MyType> &randoms);

    std::complex<MyType>
    CalculateCylinderFF(std::complex<MyType> qpar, std::complex<MyType> q, std::complex<MyType> qz,
                                       int first_parameter_index, int first_random_index,
                                       const std::vector<Vector2<MyType>> &parameters, const std::vector<MyType> &randoms);

    std::complex<MyType> Eiz(std::complex<MyType> z);
    std::complex<MyType>  Sinc(std::complex<MyType> z);
}


#endif
