#ifndef _CU_COMPLEX_NUMERIC_CUH_
#define _CU_COMPLEX_NUMERIC_CUH_

/*For real valued functions*/
#include <cmath>    

#include <cuComplex.h>
#include <math_constants.h>
#include "common/standard_defs.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gpu/util/util.h"
/*Switch complex type*/
//typedef my_complex my_complex;

/*Device Constants*/
__device__ const MyType CUTINY_ = 1.0e-13;
__device__ const MyType PI_ = CUDART_PI;
__device__ const MyType FOUR_PI_ = 4 * CUDART_PI;
__device__ const MyType REAL_ZERO_ = 0.f;

/*Helper functions*/
__device__ static __inline__ MyType cuRtan(MyType a){
    return tanf(a);
  }

  __device__ static __inline__ double cuRtan(double a){
    return tan(a);
  }

  __device__ static __inline__ MyComplex make_cuC(MyType r, MyType i) {
    return make_cuFloatComplex(r, i);
  } // make_cuC()

  __device__ static __inline__ cuDoubleComplex make_cuC(double r, double i) {
    return make_cuDoubleComplex(r, i);
  } // make_cuC()

  // get the real part from my_type2 type complex
  __device__ static __inline__ MyType cu_real (MyComplex x) {
    return cuCrealf (x);
  }

  // get the real part from double type complex
  __device__ static __inline__ double cu_real (cuDoubleComplex x) {
    return cuCreal (x);
  }

  // get the imaginary part from my_type2 type complex
  __device__ static __inline__ MyType cu_imag (MyComplex x) {
    return cuCimagf (x);
  }

  // get the imaginary part from double type complex
  __device__ static __inline__ double cu_imag (cuDoubleComplex x) {
    return cuCimag (x);
  }


  // single-precision conjugate
  __device__ static __inline__ MyComplex cuCconj(MyComplex x){
    return make_cuFloatComplex(x.x, -x.y);
  }

  // double-precision conjugate
  __device__ static __inline__ cuDoubleComplex cuCconj(cuDoubleComplex x){
    return make_cuDoubleComplex(x.x, -x.y);
  }

  // addition
  __device__ static __inline__ MyComplex operator+(MyComplex a, MyComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
  } // operator+()

  __device__ static __inline__ MyComplex operator+(MyComplex a, MyType b) {
    return make_cuFloatComplex(a.x + b, a.y);
  } // operator+()

  __device__ static __inline__ MyComplex operator+(MyType a, MyComplex b) {
    return make_cuFloatComplex(a + b.x, b.y);
  } // operator+()

  __device__ static __inline__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
  } // operator+()

  __device__ static __inline__ cuDoubleComplex operator+(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x + b, a.y);
  } // operator+()

  __device__ static __inline__ cuDoubleComplex operator+(double a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a + b.x, b.y);
  } // operator+()


  // subtraction

  __device__ static __inline__ MyComplex operator-(MyComplex a, MyComplex b) {
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
  } // operator-()

  __device__ static __inline__ MyComplex operator-(MyComplex a, MyType b) {
    return make_cuFloatComplex(a.x - b, a.y);
  } // operator-()

  __device__ static __inline__ MyComplex operator-(MyType a, MyComplex b) {
    return make_cuFloatComplex(a - b.x, - b.y);
  } // operator-()

  __device__ static __inline__ cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
  } // operator-()

  __device__ static __inline__ cuDoubleComplex operator-(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x - b, a.y);
  } // operator-()

  __device__ static __inline__ cuDoubleComplex operator-(double a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a - b.x, - b.y);
  } // operator-()


  // multiplication

  __device__ static __inline__ MyComplex operator*(MyComplex a, MyComplex b) {
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  } // operator*()

  __device__ static __inline__ MyComplex operator*(MyComplex a, MyType b) {
    return make_cuFloatComplex(a.x * b, a.y * b);
  } // operator*()

  __device__ static __inline__ MyComplex operator*(MyType a, MyComplex b) {
    return make_cuFloatComplex(a * b.x, a * b.y);
  } // operator*()

  __device__ static __inline__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  } // operator*()

  __device__ static __inline__ cuDoubleComplex operator*(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x * b, a.y * b);
  } // operator*()

  __device__ static __inline__ cuDoubleComplex operator*(double a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a * b.x, a * b.y);
  } // operator*()


  // division

  __device__ static __inline__ MyComplex operator/(MyComplex a, MyComplex b) {
    return cuCdivf(a, b);
  } // operator/()

  __device__ static __inline__ cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCdiv(a, b);
  } // operator/()

  __device__ static __inline__ MyComplex operator/(MyComplex a, MyType b) {
    return make_cuFloatComplex(a.x / b, a.y / b);
  } // operator/()

  __device__ static __inline__ cuDoubleComplex operator/(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x / b, a.y / b);
  } // operator/()

  __device__ static __inline__ MyComplex operator/(MyType a, MyComplex b) {
    //return cuCdivf(make_my_complex(a, 0.0f), b);
    MyType den = b.x * b.x + b.y * b.y;
    MyType den_inv = 1.0f / den;
    return make_cuFloatComplex((a * b.x) * den_inv, (a * b.y) * den_inv);
  } // operator/()

  __device__ static __inline__ cuDoubleComplex operator/(double a, cuDoubleComplex b) {
    return cuCdiv(make_cuDoubleComplex(a, 0.0), b);
  } // operator/()

  __device__ static __inline__ MyType cuC_abs(MyComplex a) {
    return sqrtf(a.x * a.x + a.y * a.y);
  } // cuC_abs()

  __device__ static __inline__ double cuC_abs(cuDoubleComplex a) {
    return sqrt(a.x * a.x + a.y * a.y);
  } // cuC_abs()

  __device__ static __inline__ MyType cuC_norm(MyComplex a){
    return (a.x * a.x + a.y * a.y);
  } // cuC_norm()

  __device__ static __inline__ double cuC_norm(cuDoubleComplex a){
    return (a.x * a.x + a.y * a.y);
  } // cuC_norm()

  // rotate the Q-vector my_type2
  __device__ static __inline__ void rotate_q(MyType * rot, MyType qx, MyType qy, MyComplex qz,
          MyComplex & mqx, MyComplex & mqy, MyComplex & mqz) {
      mqx.x = rot[0] * qx + rot[1] * qy + rot[2] * qz.x; mqx.y = rot[2] * qz.y;
      mqy.x = rot[3] * qx + rot[4] * qy + rot[5] * qz.x; mqy.y = rot[5] * qz.y;
      mqz.x = rot[6] * qx + rot[7] * qy + rot[8] * qz.x; mqz.y = rot[8] * qz.y;
  }

  // rotate the Q-vector double
  __device__ static __inline__ void rotate_q(double * rot, double qx, double qy, 
          cuDoubleComplex qz, cuDoubleComplex & mqx, cuDoubleComplex & mqy, cuDoubleComplex & mqz){

      mqx.x = rot[0] * qx + rot[1] * qy + rot[2] * qz.x; mqx.y = rot[2] * qz.y;
      mqy.x = rot[3] * qx + rot[4] * qy + rot[5] * qz.x; mqy.y = rot[5] * qz.y;
      mqz.x = rot[6] * qx + rot[7] * qy + rot[8] * qz.x; mqz.y = rot[8] * qz.y;
  }

  __device__ static __inline__ MyComplex cuCsqrt(MyComplex z) {
    MyType x = z.x;
    MyType y = z.y;
    if (x == 0) {
      MyType t = sqrtf(fabsf(y) / 2);
      return make_cuC(t, y < 0 ? -t : t);
    } else if (y == 0) {
      MyType t = sqrtf(fabsf(x));
      return x < 0 ? make_cuC(0, t) : make_cuC(t, 0);
    } else {
      //my_type2 t = sqrtf(2 * (cuCabsf(z) + fabsf(x)));
      MyType t = sqrtf(2 * (sqrtf(x * x + y * y) + fabsf(x)));
      MyType u = t / 2;
      return x > 0 ? make_cuC(u, y / t) : make_cuC(fabsf(y) / t, y < 0 ? -u : u);
    } // if-else
  } // cuCsqrt()

 //__device__ static __inline__ MyComplex cuCsqrt(MyComplex x)
 // {
 //     MyType radius = cuCabs(x);
 //     MyType cosA = x.x / radius;
 //     MyComplex out;
 //     out.x = sqrt(radius * (cosA + 1.0) / 2.0);
 //     out.y = sqrt(radius * (1.0 - cosA) / 2.0);
 //     // signbit should be false if x.y is negative
 //     if (signbit(x.y))
 //         out.y *= -1.0;

 //     return out;
 // }

  //__device__ static __inline__ cuDoubleComplex cuCsqrt(cuDoubleComplex z) {
  //  double x = z.x;
  //  double y = z.y;
  //  if (x == 0) {
  //    double t = sqrt(fabs(y) / 2);
  //    return make_cuC(t, y < 0 ? -t : t);
  //  } else if (y == 0) {
  //    return x < 0 ? make_cuC(0, sqrt(fabs(x))) : make_cuC(sqrt(x), 0);
  //  } else {
  //    //double t = sqrt(2 * (cuCabs(z) + fabs(x)));
  //    double t = sqrt(2 * (sqrt(x * x + y * y) + fabs(x)));
  //    double u = t / 2;
  //    return x > 0 ? make_cuC(u, y / t) : make_cuC(fabs(y) / t, y < 0 ? -u : u);
  //  } // if-else
  //} // cuCsqrt()

  __device__ static __inline__ MyComplex cuCsqr(MyComplex a) {
    return make_cuFloatComplex(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
  } // cuCsqr()

  __device__ static __inline__ cuDoubleComplex cuCsqr(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y);
  } // cuCsqr()

  // eucledian norm
  __device__ static __inline__ MyType cuCnorm3(MyComplex * a){
    MyType res = 0.f;
    for (int i = 0; i < 3; i++) res += cuC_norm(a[i]);
    return res;
  } // cuCnorm3()

  __device__ static __inline__ double cuCnorm3(cuDoubleComplex * a){
    MyType res = 0.;
    for (int i = 0; i < 3; i++) res += cuC_norm(a[i]);
    return res;
  }
  __device__ static __inline__ MyType cuCnorm3(MyComplex a, MyComplex b, MyComplex c) {
    MyType a2 = cuC_norm(a);
    MyType b2 = cuC_norm(b);
    MyType c2 = cuC_norm(c);
    return (a2 + b2 + c2);
  } // cuCnorm3()

  __device__ static __inline__ double cuCnorm3(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    double a2 = cuC_norm(a);
    double b2 = cuC_norm(b);
    double c2 = cuC_norm(c);
    return (a2 + b2 + c2);
  } // cuCnorm3()

  __device__ static __inline__ MyType cuCnorm3(MyType a, MyType b, MyComplex c) {
    MyType c2 = cuC_norm(c);
    return (a * a + b * b + c2);
  } // cuCnorm3()

  __device__ static __inline__ double cuCnorm3(double a, double b, cuDoubleComplex c) {
    double c2 = cuC_norm(c);
    return (a * a + b * b + c2);
  } // cuCnorm3()

  // e^z = e^z.x (cos(z.y) + isin(z.y))
  __device__ static __inline__ MyComplex cuCexp(MyComplex z) {
    MyType temp1 = cosf(z.y);
    MyType temp2 = sinf(z.y);
    MyType temp3 = expf(z.x);
    return make_cuFloatComplex(temp1 * temp3, temp2 * temp3);
  } // cuCexp()

  __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex z) {
    double temp1 = cos(z.y);
    double temp2 = sin(z.y);
    double temp3 = exp(z.x);
    return make_cuDoubleComplex(temp1 * temp3, temp2 * temp3);
  } // cuCexp()

  // e^if = cos(f) + isin(f)
  __device__ static __inline__ MyComplex cuCexpi(MyType f) {
    return make_cuFloatComplex(cosf(f), sinf(f));
  } // cuCexpi()

  __device__ static __inline__ cuDoubleComplex cuCexpi(double f) {
    return make_cuDoubleComplex(cos(f), sin(f));
  } // cuCexpi()

  // e^iz
  __device__ static __inline__ MyComplex cuCexpi(MyComplex z) {
    return cuCexp(make_cuFloatComplex(-z.y, z.x));
  } // cuCexpi()

  __device__ static __inline__ cuDoubleComplex cuCexpi(cuDoubleComplex z) {
    return cuCexp(make_cuDoubleComplex(-z.y, z.x));
  } // cuCexpi()

  __device__ static __inline__ MyType cuCabs(MyComplex z) {
    return hypotf(z.x, z.y);
  } // cuCabs()

  __device__ static __inline__ MyType cuCarg(MyComplex z) {
    return atan2f(z.y, z.x);
  } // cuCarg()

  __device__ static __inline__ double cuCarg(cuDoubleComplex z) {
    return atan2(z.y, z.x);
  } // cuCarg()

  __device__ static __inline__ MyComplex cuClog(MyComplex z) {
    return make_cuFloatComplex(log(cuCabs(z)), cuCarg(z));
  } // cuClog()

  __device__ static __inline__ cuDoubleComplex cuClog(cuDoubleComplex z) {
    return make_cuDoubleComplex(log(cuCabs(z)), cuCarg(z));
  } // cuClog()

  // __device__ static __inline__ MyComplex cuCpow(MyComplex a, MyType2 p) {
  //   return cuCexp(cuClog(a) * p);;
  // } // cuCpow()

  __device__ static __inline__ cuDoubleComplex cuCpow(cuDoubleComplex a, double p) {
    return cuCexp(cuClog(a) * p);;
  } // cuCpow()

  __device__ static __inline__ MyComplex cuCpow(MyComplex a, MyType p) {
      return cuCexp(cuClog(a) * p);;
  } // cuCpow()

  __device__ static __inline__ MyComplex cuCpow(MyComplex a, MyComplex p) {
    return cuCexp(cuClog(a) * p);;
  } // cuCpow()

  __device__ static __inline__ cuDoubleComplex cuCpow(cuDoubleComplex a, cuDoubleComplex p) {
    return cuCexp(cuClog(a) * p);
  } // cuCpow()

  __device__ static __inline__ MyComplex cuCsin(MyComplex z) {
    MyType x = z.x;
    MyType y = z.y;
    return make_cuC(sinf(x) * coshf(y), cosf(x) * sinhf(y));
  } // cuCsin()

  __device__ static __inline__ cuDoubleComplex cuCsin(cuDoubleComplex z) {
    double x = z.x;
    double y = z.y;
    return make_cuC(sin(x) * cosh(y), cos(x) * sinh(y));
  } // cuCsin()

  __device__ static __inline__ MyComplex cuCcos(MyComplex z) {
    MyType x = z.x;
    MyType y = z.y;
    return make_cuC(cosf(x) * coshf(y), -sinf(x) * sinhf(y));
  } // cuCsin()

  __device__ static __inline__ cuDoubleComplex cuCcos(cuDoubleComplex z) {
    double x = z.x;
    double y = z.y;
    return make_cuC(cos(x) * cosh(y), -sin(x) * sinh(y));
  } // cuCsin()

  __device__ static __inline__ MyType cuCsinc(MyType v) {
      if (fabs(v) > 1.0E-05)
        return (sin(v)/v);
      else
        return (1. - v*v/6.f + v*v*v*v/120.f);
  }

  __device__ static __inline__ double cuCsinc(double v) {
      if (abs(v) > 1.0E-05)
        return (sin(v)/v);
      else
        return (1. - v*v/6. + v*v*v*v/120.);
  }

  __device__ static __inline__ MyComplex cuCsinc(MyComplex value) {
    MyComplex temp;
    //if(fabsf(value.x) < 1e-14 && fabsf(value.y) < 1e-14) temp = make_my_complex(1.0, 0.0);
    if(fabsf(value.x) < 1e-5 && fabsf(value.y) < 1e-5) temp = make_cuFloatComplex(1.0, 0.0);
    else temp = cuCsin(value) / value;
    return temp;
  } // cuCsinc()

  __device__ static __inline__ cuDoubleComplex cuCsinc(cuDoubleComplex value) {
    cuDoubleComplex temp;
    //if(fabs(value.x) < 1e-14 && fabs(value.y) < 1e-14) temp = make_cuDoubleComplex(1.0, 0.0);
    if(fabs(value.x) < 1e-9 && fabs(value.y) < 1e-9) temp = make_cuDoubleComplex(1.0, 0.0);
    else temp = cuCsin(value) / value;
    return temp;
  } // cuCsinc()

  __device__ static __inline__ bool cuCiszero(MyComplex value) {
    //return (fabsf(value.x) < 1e-14 && fabsf(value.y) < 1e-14);
    return (fabsf(value.x) < 1e-5 && fabsf(value.y) < 1e-5);
  } // cuCiszero()

  __device__ static __inline__ bool cuCiszero(cuDoubleComplex value) {
    //return (fabs(value.x) < 1e-14 && fabs(value.y) < 1e-14);
    return (fabs(value.x) < 1e-9 && fabs(value.y) < 1e-9);
  } // cuCiszero()

  __device__ static __inline__ MyComplex cuCcbessj(MyComplex zz, int order) {
      return make_cuFloatComplex(j1(cu_real(zz)), 0.0);
  }
#endif