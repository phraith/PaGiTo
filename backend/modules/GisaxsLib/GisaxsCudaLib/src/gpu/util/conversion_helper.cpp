//
// Created by Phil on 06.03.2022.
//

#include "gpu/util/conversion_helper.h"

std::vector <MyComplex> GpuConversionHelper::Convert(const std::vector<std::complex<MyType>> &input)
{
    std::vector<MyComplex> converted_vector;
    for (const auto &element : input) {
        converted_vector.emplace_back(MyComplex {element.real(), element.imag()});
    }

    return converted_vector;
}

MyType2 GpuConversionHelper::Convert(const Vector2<MyType> &input) {
    return MyType2 {input.x, input.y};
}

MyType2I GpuConversionHelper::Convert(const Vector2<int> &input) {
    return MyType2I {input.x, input.y };
}

std::vector<MyType2> GpuConversionHelper::Convert(const std::vector<Vector2<MyType>> &input) {
    std::vector<MyType2> converted_vector;
    for (const auto &element : input) {
        converted_vector.emplace_back(MyComplex {element.x, element.y});
    }

    return converted_vector;
}

std::vector<MyType3> GpuConversionHelper::Convert(const std::vector<Vector3<MyType>> &input) {
    std::vector<MyType3> converted_vector;
    for (const auto &element : input) {
        converted_vector.emplace_back(MyType3 {element.x, element.y, element.z});
    }

    return converted_vector;
}

MyType3I GpuConversionHelper::Convert(const Vector3<int> &input) {
    return MyType3I {input.x, input.y, input.z};
}

MyType3 GpuConversionHelper::Convert(const Vector3<MyType> &input) {
    return MyType3 {input.x, input.y, input.z};
}

MyComplex GpuConversionHelper::Convert(const std::complex<MyType> &input) {
    return {input.real(), input.imag()};
}
