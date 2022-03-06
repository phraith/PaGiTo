//
// Created by Phil on 06.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SHAPE_H
#define GISAXSMODELINGFRAMEWORK_SHAPE_H

#include <cuda_runtime.h>
#include "gpu/util/util.h"

class ShapeFF
{
public:
    virtual __device__ ~ShapeFF() {};

    virtual __device__ MyComplex Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx) = 0;
    virtual __device__ MyComplex Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx) = 0;

    virtual __device__ ShapeTypeV2 Type() = 0;
    virtual __device__ int ParamCount() = 0;
private:
};

#endif //GISAXSMODELINGFRAMEWORK_SHAPE_H
