#ifndef MODULES_COMMON_INC_STANDARD_DEFS_H
#define MODULES_COMMON_INC_STANDARD_DEFS_H

#include <vector>
#include <string>
#include "nlohmann/json.hpp"

typedef float MyType;

enum class ShapeTypeV2 {
    sphere=0, cylinder=1
};

template <typename T>
struct Vector2{
    T x;
    T y;
};

template <typename T>
struct Vector3{
    T x;
    T y;
    T z;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ShapeTypeV2, {
{ ShapeTypeV2::sphere, "sphere" },
{ ShapeTypeV2::cylinder, "cylinder" }
})

enum class ConstantMemoryId {
    QGRID_XY = 0,
    QGRID_Z = 1,
    QGRID_QPAR = 2,
    QGRID_Q = 3,
    QGRID_COEFFS= 4
};

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

    Vector2<int> resolution;

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