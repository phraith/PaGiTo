//
// Created by Phil on 06.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_CONVERSION_HELPER_H
#define GISAXSMODELINGFRAMEWORK_CONVERSION_HELPER_H

#include <vector>
#include <complex>
#include "util.h"
#include "cuda_numerics.h"

namespace GpuConversionHelper {
    std::vector<MyComplex> Convert(const std::vector<std::complex<MyType>> &input);
    std::vector<MyType2> Convert(const std::vector<Vector2<MyType>> &input);
    std::vector<MyType3> Convert(const std::vector<Vector3<MyType>> &input);

    MyType2 Convert(const Vector2<MyType> &input);
    MyType2I Convert(const Vector2<int> &input);
    MyType3I Convert(const Vector3<int> &input);
    MyType3 Convert(const Vector3<MyType> &input);
}


#endif //GISAXSMODELINGFRAMEWORK_CONVERSION_HELPER_H
