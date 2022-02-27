#include "gpu/ff/sphere.h"

#include "gpu/util/cuda_numerics.h"

#include "stdio.h"
#include <common/standard_constants.h>

__device__ SphereFF::SphereFF(MyType2 radius, int rand_count)
    :
    radius_(radius),
    rand_count_(rand_count),
    rand_rad_(new MyType [rand_count_]),
    type_(ShapeTypeV2::sphere)
{
}

__device__ SphereFF::~SphereFF()
{
    if (rand_rad_ != nullptr)
        delete[] rand_rad_;
}

__device__ MyComplex SphereFF::Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx)
{
    MyComplex q = cuCsqrt(qx * qx + qy * qy + qz * qz);

    MyType radius = rand_rad_[rand_idx];

    MyComplex qR = q * radius;

    if (cuC_abs(qR) < CUTINY_) {
        return { 0,0 };
    }

    MyComplex sincos = cuCsin(qR) - cuCcos(qR) * qR;
    MyComplex expval = cuCexpi(qz * radius);

    MyComplex ff = (FOUR_PI_ / (q * q * q)) * expval * sincos;
    return ff;
}

__device__ MyComplex SphereFF::Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx)
{
    MyType radius = rand_rad_[rand_idx];

    MyComplex qR = q * radius;

    if (cuC_abs(qR) < CUTINY_) {
        return {0,0};
    }

    MyComplex sincos = cuCsin(qR) - cuCcos(qR) * qR;
    MyComplex expval = cuCexpi(qz * radius);

    MyComplex ff = (FOUR_PI_ / (q * q * q)) * expval * sincos;
    return ff;
}

__device__ ShapeTypeV2 SphereFF::Type()
{
    return type_;
}

__device__ int SphereFF::ParamCount()
{
    return 1;
}

__device__ MyType* SphereFF::RandRads()
{
    return rand_rad_;
}

__device__ MyType2 SphereFF::Radius()
{
    return radius_;
}

__device__ void SphereFF::Update(MyType2 new_radius)
{
    radius_ = new_radius;
}