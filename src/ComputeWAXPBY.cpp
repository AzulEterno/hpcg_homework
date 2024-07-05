
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

#include "immintrin.h"

#include "config.hpp"
#include <iostream>

int ComputeWAXPBY_Intel_AVX_zcy(const local_int_t n, const double alpha, const Vector& x,
    const double beta, const Vector& y, Vector& w);

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector& x,
    const double beta, const Vector& y, Vector& w, bool& isOptimized) {

  if (g_optimization_type == OPTIM_TYPE_REF) {
    std::cout << "ComputeWAXPBY_ref" << std::endl;
    return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
  }
  else if (g_optimization_type == OPTIM_TYPE_ZCY) {
    // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
    //isOptimized = false;
    std::cout << "ComputeWAXPBY_zcy" << std::endl;
    return ComputeWAXPBY_Intel_AVX_zcy(n, alpha, x, beta, y, w);
  }
  else {
    std::cout << "ComputeWAXPBY_ref" << std::endl;
    return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
  }
}


//#include "ComputeWAXPBY_Intel_AVX.hpp"
 /*!
   Routine to compute the update of a vector with the sum of two
   scaled vectors where: w = alpha*x + beta*y

   This routine calls the reference WAXPBY implementation by default, but
   can be replaced by a custom, optimized routine suited for
   the target system.

   @param[in] n the number of vector elements (on this processor)
   @param[in] alpha, beta the scalars applied to x and y respectively.
   @param[in] x, y the input vectors
   @param[out] w the output vector
   @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

   @return returns 0 upon success and non-zero otherwise

 */

int ComputeWAXPBY_Intel_AVX_zcy(const local_int_t n, const double alpha, const Vector& x,
    const double beta, const Vector& y, Vector& w) {

  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  const double* const xv = x.values;
  const double* const yv = y.values;
  double* const wv = w.values;

#if defined(AVX512_Optimization)
  const local_int_t vector_width = 8; // AVX512 processes 8 doubles at a time
#elif defined(AVX2_Optimization)
  const local_int_t vector_width = 4; // AVX2 processes 4 doubles at a time
#else
  const local_int_t vector_width = 1; // Scalar processing
#endif



  if (alpha == 1.0) {

    local_int_t i = 0;
#if defined(AVX512_Optimization) || defined(AVX2_Optimization)
#if defined(AVX512_Optimization)
    //__m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
#elif defined(AVX2_Optimization)
    //__m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);
#endif
    for (; i <= n - vector_width; i += vector_width) {
#if defined(AVX512_Optimization)
      __m512d xv_vec = _mm512_loadu_pd(&xv[i]);
      __m512d yv_vec = _mm512_loadu_pd(&yv[i]);
      //__m512d beta_vec = _mm512_set1_pd(beta);
      __m512d result = _mm512_fmadd_pd(beta_vec, yv_vec, xv_vec);
      _mm512_storeu_pd(&wv[i], result);
#elif defined(AVX2_Optimization)
      __m256d xv_vec = _mm256_loadu_pd(&xv[i]);
      __m256d yv_vec = _mm256_loadu_pd(&yv[i]);
      //__m256d beta_vec = _mm256_set1_pd(beta);
      __m256d result = _mm256_fmadd_pd(beta_vec, yv_vec, xv_vec);
      _mm256_storeu_pd(&wv[i], result);
#endif
    }
#endif

    //Serial Part
    for (; i < n; ++i) {
      wv[i] = xv[i] + beta * yv[i];
    }
  }
  else if (beta == 1.0) {
    local_int_t i = 0;
#if defined(AVX512_Optimization) || defined(AVX2_Optimization)
#if defined(AVX512_Optimization)
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    //__m512d beta_vec = _mm512_set1_pd(beta);
#elif defined(AVX2_Optimization)
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    //__m256d beta_vec = _mm256_set1_pd(beta);
#endif
    for (; i <= n - vector_width; i += vector_width) {
#if defined(AVX512_Optimization)
      __m512d xv_vec = _mm512_loadu_pd(&xv[i]);
      __m512d yv_vec = _mm512_loadu_pd(&yv[i]);
      //__m512d alpha_vec = _mm512_set1_pd(alpha);
      __m512d result = _mm512_fmadd_pd(alpha_vec, xv_vec, yv_vec);
      _mm512_storeu_pd(&wv[i], result);
#elif defined(AVX2_Optimization)
      __m256d xv_vec = _mm256_loadu_pd(&xv[i]);
      __m256d yv_vec = _mm256_loadu_pd(&yv[i]);
      //__m256d alpha_vec = _mm256_set1_pd(alpha);
      __m256d result = _mm256_fmadd_pd(alpha_vec, xv_vec, yv_vec);
      _mm256_storeu_pd(&wv[i], result);
#endif
    }
#endif
    //Serial Part
    for (; i < n; ++i) {
      wv[i] = alpha * xv[i] + yv[i];
    }
  }
  else {
    local_int_t i = 0;
#if defined(AVX512_Optimization) || defined(AVX2_Optimization)
#if defined(AVX512_Optimization)
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
#elif defined(AVX2_Optimization)
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);
#endif
    for (; i <= n - vector_width; i += vector_width) {
#if defined(AVX512_Optimization)
      __m512d xv_vec = _mm512_loadu_pd(&xv[i]);
      __m512d yv_vec = _mm512_loadu_pd(&yv[i]);

      __m512d result = _mm512_add_pd(_mm512_mul_pd(alpha_vec, xv_vec), _mm512_mul_pd(beta_vec, yv_vec));
      _mm512_storeu_pd(&wv[i], result);
#elif defined(AVX2_Optimization)
      __m256d xv_vec = _mm256_loadu_pd(&xv[i]);
      __m256d yv_vec = _mm256_loadu_pd(&yv[i]);

      __m256d result = _mm256_add_pd(_mm256_mul_pd(alpha_vec, xv_vec), _mm256_mul_pd(beta_vec, yv_vec));
      _mm256_storeu_pd(&wv[i], result);
#endif
    }
#endif
    //Serial Part
    for (; i < n; ++i) {
      wv[i] = alpha * xv[i] + beta * yv[i];
    }
  }

  return 0;
}
