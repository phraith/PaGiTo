#ifndef MODULES_COMMON_INC_CUDA_TYPES_STANDARD_VECTOR_TYPES_H
#define MODULES_COMMON_INC_CUDA_TYPES_STANDARD_VECTOR_TYPES_H

#include "vector_types.h"
#include <cuComplex.h>

typedef float MyType;
typedef float2 MyType2;
typedef float3 MyType3;
typedef float4 MyType4;

typedef cuFloatComplex MyComplex;

typedef int2 MyType2I;
typedef int3 MyType3I;
typedef int4 MyType4I;

typedef unsigned int MyUint;

#endif