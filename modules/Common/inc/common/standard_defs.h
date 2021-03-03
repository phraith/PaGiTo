#ifndef MODULES_COMMON_INC_STANDARD_DEFS_H
#define MODULES_COMMON_INC_STANDARD_DEFS_H

#include <vector>
#include <string>

#include "standard_vector_types.h"

enum class ShapeType { kSphere = 0, kCylinder = 1, kTrapezoid = 2};

static std::string ShapeTypeToString(ShapeType type)
{
    switch (type)
    {
    case ShapeType::kSphere:
        return "sphere";
    case ShapeType::kCylinder:
        return "cylinder";
    case ShapeType::kTrapezoid:
        return "trapezoid";
    default:
        return "";
    }
}


typedef struct MyComplex4
{
    MyComplex x;
    MyComplex y;
    MyComplex z;
    MyComplex w;
} MyComplex4;

typedef struct ParamData
{
    double value;
    double stddev;
} ParamData;

typedef struct SimData
{
	MyType fitness;
	std::vector<MyType> intensities;
    std::vector<MyType> qx;
    std::vector<MyType> qy;
    std::vector<MyType> qz;

    MyType2I resolution;

    float scale;
} SimData;

typedef struct TimeMeasurement
{
    std::string device_name;

    double kernel_time;
    double full_runtime;

    double average_kernel_runtime;
    double average_full_runtime;

    int runs;
} TimeMeasurement;

class ShapeFF
{
public:
    virtual __device__ ~ShapeFF() {};

    virtual __device__ MyComplex Evaluate(MyComplex qx, MyComplex qy, MyComplex qz, int rand_idx) = 0;
    virtual __device__ MyComplex Evaluate2(MyComplex qpar, MyComplex q, MyComplex qz, int rand_idx) = 0;

    virtual __device__ ShapeType Type() = 0;
    virtual __device__ int ParamCount() = 0;
private:
};

enum class WorkStatus
{
    kIdle,
    kWorking
};

typedef struct FittedParameter
{
    double value;
    double stddev;

    std::string type;
}
FittedParameter;

typedef struct FittedShape
{
    std::string type;
    std::vector<FittedParameter> parameters;
}FittedShape;



#endif