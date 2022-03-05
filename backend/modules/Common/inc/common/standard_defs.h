#ifndef MODULES_COMMON_INC_STANDARD_DEFS_H
#define MODULES_COMMON_INC_STANDARD_DEFS_H

#include <vector>
#include <string>
#include "nlohmann/json.hpp"
#include "standard_vector_types.h"

//enum class ShapeType { kSphere = 0, kCylinder = 1, kTrapezoid = 2};
enum class ShapeTypeV2 {
    sphere=0, cylinder=1
};

NLOHMANN_JSON_SERIALIZE_ENUM(ShapeTypeV2, {
{ ShapeTypeV2::sphere, "sphere" },
{ ShapeTypeV2::cylinder, "cylinder" }
})

//static std::string ShapeTypeToString(ShapeType type)
//{
//    switch (type)
//    {
//    case ShapeType::kSphere:
//        return "sphere";
//    case ShapeType::kCylinder:
//        return "cylinder";
//    case ShapeType::kTrapezoid:
//        return "trapezoid";
//    default:
//        return "";
//    }
//}

enum class ConstantMemoryId {
    QGRID_XY = 0,
    QGRID_Z = 1,
    QGRID_QPAR = 2,
    QGRID_Q = 3,
    QGRID_COEFFS= 4
};

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
    std::vector<unsigned char> normalized_intensities;
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

    virtual __device__ ShapeTypeV2 Type() = 0;
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