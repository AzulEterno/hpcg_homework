
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "immintrin.h"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "Vector.hpp"
#include <iostream>
#include "config.hpp"

// Helper function to reduce an AVX2 vector to a single scalar value
inline double _mm256_reduce_add_pd(__m256d vec) {
  __m128d hi = _mm256_extractf128_pd(vec, 1);
  __m128d lo = _mm256_castpd256_pd128(vec);
  __m128d sum = _mm_add_pd(lo, hi);
  sum = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));
  return _mm_cvtsd_f64(sum);
}

int ComputeDotProduct_Intel_AVX_zcy(const local_int_t n, const Vector& x, const Vector& y,
    double& result, double& time_allreduce) {
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  const double* xv = x.values;
  const double* yv = y.values;

#if defined(AVX512_Optimization)
  const local_int_t vector_width = 8; // AVX512 processes 8 doubles at a time
#elif defined(AVX2_Optimization)
  const local_int_t vector_width = 4; // AVX2 processes 4 doubles at a time
#else
  const local_int_t vector_width = 1; // Scalar processing
#endif

  local_int_t i = 0;

  if (yv == xv) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
    for (i = 0; i <= n - vector_width; i += vector_width) {
#if defined(AVX512_Optimization)
      __m512d xv_vec = _mm512_loadu_pd(&xv[i]);
      __m512d result_vec = _mm512_mul_pd(xv_vec, xv_vec);
      local_result += _mm512_reduce_add_pd(result_vec);
#elif defined(AVX2_Optimization)
      __m256d xv_vec = _mm256_loadu_pd(&xv[i]);
      __m256d result_vec = _mm256_mul_pd(xv_vec, xv_vec);
      local_result += _mm256_reduce_add_pd(result_vec);
#else
      local_result += xv[i] * xv[i];
#endif
    }
  }
  else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
    for (i = 0; i <= n - vector_width; i += vector_width) {
#if defined(AVX512_Optimization)
      __m512d xv_vec = _mm512_loadu_pd(&xv[i]);
      __m512d yv_vec = _mm512_loadu_pd(&yv[i]);
      __m512d result_vec = _mm512_mul_pd(xv_vec, yv_vec);
      local_result += _mm512_reduce_add_pd(result_vec);
#elif defined(AVX2_Optimization)
      __m256d xv_vec = _mm256_loadu_pd(&xv[i]);
      __m256d yv_vec = _mm256_loadu_pd(&yv[i]);
      __m256d result_vec = _mm256_mul_pd(xv_vec, yv_vec);
      local_result += _mm256_reduce_add_pd(result_vec);
#else
      local_result += xv[i] * yv[i];
#endif
    }
  }

  // Handle any remaining elements
  for (; i < n; ++i) {
    if (yv == xv) {
      local_result += xv[i] * xv[i];
    }
    else {
      local_result += xv[i] * yv[i];
    }
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = false;
  // return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
  // std::cout << "ComputeDotProduct" << std::endl;
  assert(g_dot_product_type == DOT_PRODUCT_TYPE_REF || g_dot_product_type == DOT_PRODUCT_TYPE_ZCY);
  
  if (g_dot_product_type == DOT_PRODUCT_TYPE_REF) {

#if defined(DebugPrintExecuteCalls)
    std::cout << "ComputeDotProduct_ref" << std::endl;
#endif
    return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
  }
  else {
    // g_dot_product_type is DOT_PRODUCT_TYPE_ZCY
#if defined(DebugPrintExecuteCalls)
    std::cout << "ComputeDotProduct_zcy" << std::endl;
#endif
    return ComputeDotProduct_Intel_AVX_zcy(n, x, y, result, time_allreduce);
  }
}
