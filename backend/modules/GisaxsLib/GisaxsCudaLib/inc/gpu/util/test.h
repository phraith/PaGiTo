#ifndef GISAXS_LIB_TEST_CUH
#define GISAXS_LIB_TEST_CUH

#include "cuda_runtime.h"
#include "standard_vector_types.h"

extern "C++"
{
    void add(int *x, int *y, int *res, int elems);

    void assign_value(MyType *out, int size, cudaStream_t work_stream);
    void schwefel(MyType* out, MyType* x, int size, cudaStream_t work_stream);

    void img_diff(MyType* out_img, MyType* real_img, MyType val, int size, cudaStream_t work_stream);
}

#endif