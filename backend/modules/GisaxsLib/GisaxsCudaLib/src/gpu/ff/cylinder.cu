#include "gpu/ff/cylinder.h"

#include "gpu/util/cuda_numerics.h"

#include "common/standard_constants.h"

#include "stdio.h"

__device__ CylinderFF::CylinderFF(MyType2 radius, MyType2 height, int rand_count)
    :
    radius_(radius),
    height_(height),
    rand_count_(rand_count),
    rand_rads_(new MyType[rand_count_]),
    rand_heights_(new MyType[rand_count_]),
    type_(ShapeType::kCylinder)
{
}

__device__ CylinderFF::~CylinderFF()
{
    if (rand_rads_ != nullptr)
        delete[] rand_rads_;

    if (rand_heights_ != nullptr)
        delete[] rand_heights_;
}

__device__ MyComplex CylinderFF::Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx)
{
    MyType radius = rand_rads_[rand_idx];
    MyType height = rand_heights_[rand_idx];

    MyComplex qpar = cuCsqrt(qx * qx + qy * qy);

    MyType     temp1 = 2 * PI_ * radius * radius * height;
    MyComplex temp2 = { 0,0 };

    if ((qpar.x * qpar.x + qpar.y * qpar.y) > CUTINY_)
        temp2 = make_cuFloatComplex(j1f(cu_real(qpar * radius)), 0.0) / (qpar * radius);

    MyComplex temp3 = cuCsinc(0.5 * qz * height);
    MyComplex temp4 = cuCexpi(0.5 * qz * height);

    MyComplex ff = temp1 * temp2 * temp3 * temp4;

    return ff;
}

__device__ MyComplex CylinderFF::Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx)
{
    MyType radius = rand_rads_[rand_idx];
    MyType height = rand_heights_[rand_idx];
    
    MyType     temp1 = 2 * PI_ * radius * radius * height;
    MyComplex temp2 = { 0,0 };

    MyComplex qr = qpar * radius;

    if ((qpar.x * qpar.x + qpar.y * qpar.y) > CUTINY_)
        temp2 = make_cuFloatComplex(j1f(cu_real(qr)), 0.0) / (qr);

    MyComplex temp3 = cuCsinc(0.5 * qz * height);
    MyComplex temp4 = cuCexpi(0.5 * qz * height);
    return temp1 * temp2 * temp3 * temp4;
}


__device__ ShapeType CylinderFF::Type()
{
    return type_;
}

__device__ int CylinderFF::ParamCount()
{
    return 2;
}

__device__ MyType* CylinderFF::RandRads()
{
    return rand_rads_;
}

__device__ MyType* CylinderFF::RandHeights()
{
    return rand_heights_;
}

__device__ MyType2 CylinderFF::Radius()
{
    return radius_;
}

__device__ void CylinderFF::Update(MyType2 new_radius, MyType2 new_height)
{
    radius_ = new_radius;
    height_ = new_height;
}

__device__ MyType2 CylinderFF::Height()
{
    return height_;
}
