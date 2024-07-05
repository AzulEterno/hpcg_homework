
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"


#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "config.hpp"

int ComputeMG_zcy(const SparseMatrix& A, const Vector& r, Vector& x);

int ComputeMG_TDG_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {
  int ierr = 0;
  if (A.mgData != 0) {
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps - 1; i++) {
      ierr += ComputeSYMGS(A, r, x);
    }
    ierr += ComputeFusedSYMGS_SPMV_zcy(A, r, x, *A.mgData->Axf);
    if (ierr != 0) return ierr;

    ierr = ComputeRestriction_zcy(A, r);
    if (ierr != 0) return ierr;

    // ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);
    ierr = ComputeMG_zcy(*A.Ac, *A.mgData->rc, *A.mgData->xc);
    if (ierr != 0) return ierr;

    ierr = ComputeProlongation_zcy(A, x);
    if (ierr != 0) return ierr;

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; i++) {
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr != 0) return ierr;

  }
  else {
    ierr = ComputeSYMGS(A, r, x);
    if (ierr != 0) return ierr;
  }
  return 0;
}


int ComputeMG_BLOCK_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {
  int ierr = 0;
  if (A.mgData != 0) {
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps; i++) {
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr != 0) return ierr;

    ierr = ComputeSPMV(A, x, *A.mgData->Axf);
    if (ierr != 0) return ierr;

    ierr = ComputeRestriction_zcy(A, r);
    if (ierr != 0) return ierr;

    // ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);
    ierr = ComputeMG_zcy(*A.Ac, *A.mgData->rc, *A.mgData->xc);
    if (ierr != 0) return ierr;

    ierr = ComputeProlongation_zcy(A, x);
    if (ierr != 0) return ierr;

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; i++) {
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr != 0) return ierr;

  }
  else {
    ierr = ComputeSYMGS(A, r, x);
    if (ierr != 0) return ierr;
  }
  return 0;
}

int ComputeMG_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {
  assert(x.localLength == A.localNumberOfColumns);

  ZeroVector(x);

  if (A.TDG) {
    return ComputeMG_TDG_zcy(A, r, x);
  }
  return ComputeMG_BLOCK_zcy(A, r, x);

}


int ComputeMG_lmb(const SparseMatrix &A, const Vector &r, Vector &x)
{

	assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

	ZeroVector(x); // initialize x to zero  x向量初始为零向量

	int ierr = 0;
	if (A.mgData != 0)
	{ // Go to next coarse level if defined
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps; ++i) // Pre-smoothing
		{

			ierr += ComputeSYMGS(A, r, x); // Ax=r,求解x
		}
		if (ierr != 0)
			return ierr;

		ierr = ComputeSPMV(A, x, *A.mgData->Axf); // Residual_1  计算A_h * x^h

		if (ierr != 0)
			return ierr;
		// Perform restriction operation using simple injection
		ierr = ComputeRestriction_ref(A, r); // Residual_2  计算 r^h = b^h - A_h * x^h
		if (ierr != 0)
			return ierr;

		// ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc); // recusion
		ierr = ComputeMG_lmb(*A.Ac, *A.mgData->rc, *A.mgData->xc); // recusion

		if (ierr != 0)
			return ierr;
		ierr = ComputeProlongation_ref(A, x); // prolongation
		if (ierr != 0)
			return ierr;
		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for (int i = 0; i < numberOfPostsmootherSteps; ++i) // Post-smoothing  后磨光
			ierr += ComputeSYMGS(A, r, x);
		if (ierr != 0)
			return ierr;
	}
	else
	{
		ierr = ComputeSYMGS(A, r, x);
		if (ierr != 0)
			return ierr;
	}
	return 0;

	// This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
	A.isMgOptimized = true;
	// return ComputeMG_ref(A, r, x);
}


/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix& A, const Vector& r, Vector& x) {
  assert(x.localLength == A.localNumberOfColumns);

  if (g_optimization_type == OPTIM_TYPE_REF) {
    std::cout << "ComputeMG_ref" << std::endl;
    return ComputeMG_ref(A, r, x);
  }
  else if (g_optimization_type == OPTIM_TYPE_ZCY) {
    std::cout << "ComputeMG_zcy" << std::endl;
    return ComputeMG_zcy(A, r, x);
  }
  else {
    std::cout << "ComputeMG_lmb" << std::endl;
    return ComputeMG_lmb(A, r, x);
  }

}
