
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include <cassert>
#include <stdio.h>

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "a_contral.hpp"
#include "config.hpp"
#include <iostream>

int ComputeSPMV_lmb(const SparseMatrix& A, Vector& x, Vector& y);
int ComputeSPMV_zcy(const SparseMatrix& A, Vector& x, Vector& y);

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV(const SparseMatrix& A, Vector& x, Vector& y) {
	if (g_optimization_type == OPTIM_TYPE_LMB) {
#if defined(DebugPrintExecuteCalls)
		std::cout << "ComputeSPMV_lmb" << std::endl;
#endif
		return ComputeSPMV_lmb(A, x, y);
	}
	else if (g_optimization_type == OPTIM_TYPE_ZCY) {
#if defined(DebugPrintExecuteCalls)
		std::cout << "ComputeSPMV_zcy" << std::endl;
#endif
		return ComputeSPMV_zcy(A, x, y);
	}
	else {
#if defined(DebugPrintExecuteCalls)
		std::cout << "ComputeSPMV_ref" << std::endl;
#endif
		// This line and the next two lines should be removed and your version of ComputeSPMV should be used.
		A.isSpmvOptimized = false;
		return ComputeSPMV_ref(A, x, y);
	}
}

int ComputeSPMV_lmb(const SparseMatrix& A, Vector& x, Vector& y)
{

	// 并行稀疏矩阵乘法
	assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
	assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x); // 交换边界信息
#endif

	const double* const xv = x.values;			  // 向量x的值，
	double* const yv = y.values;				  // y的值
	const local_int_t nrow = A.localNumberOfRows; // 矩阵A的行数；local_int_t 为 int 类型

#ifdef HPCG_USE_ELL // 使用ELL格式优化

#ifndef HPCG_USE_MULTICOLORING_RRARRANGE // 不使用重排

	double const* cur_vals = &A.ellVal[0];		// 矩阵的值
	local_int_t const* cur_idx = &A.ellCols[0]; // 矩阵的列索引

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	// A：矩阵  x：向量   Y：结果矩阵    y = Ax
	for (local_int_t i = 0; i < nrow; i++) // 遍历矩阵行
	{
		double sum = 0.0;
		for (int j = 0; j < ELL_SIZE; j++) // 遍历矩阵非0元素
		{
			if (cur_idx[i * ELL_SIZE + j] == -1)
				break;
			sum += cur_vals[i * ELL_SIZE + j] * xv[cur_idx[i * ELL_SIZE + j]];
		}
		yv[i] = sum;
	}

	// This line and the next two lines should be removed and your version of ComputeSPMV should be used.
	A.isSpmvOptimized = true;

#else // 使用重排

	double const* cur_vals = &A.ellVal[0];		// 矩阵的值
	local_int_t const* cur_idx = &A.ellCols[0]; // 矩阵的列索引

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	// A：矩阵  x：向量   Y：结果矩阵    y = Ax
	for (local_int_t i = 0; i < nrow; i++) // 遍历矩阵行
	{
		local_int_t notRearrange = A.toNotRearrange[i]; // 获取未重排的索引
		double sum = 0.0;
		for (int j = 0; j < ELL_SIZE; j++) // 遍历矩阵非0元素
		{
			if (cur_idx[i * ELL_SIZE + j] == -1)
				break;
			sum += cur_vals[i * ELL_SIZE + j] * xv[cur_idx[i * ELL_SIZE + j]];
		}
		yv[notRearrange] = sum;
	}

	// This line and the next two lines should be removed and your version of ComputeSPMV should be used.
	A.isSpmvOptimized = true;

#endif

#else // 不使用ELL格式

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	for (local_int_t i = 0; i < nrow; i++)
	{
		double sum = 0.0;
		const double* const cur_vals = A.matrixValues[i];
		const local_int_t* const cur_inds = A.mtxIndL[i];
		const int cur_nnz = A.nonzerosInRow[i];

		for (int j = 0; j < cur_nnz; j++)
			sum += cur_vals[j] * xv[cur_inds[j]];
		yv[i] = sum;
	}
#endif

	return 0;
	// return ComputeSPMV_ref(A, x, y);
}

int ComputeSPMV_zcy(const SparseMatrix& A, Vector& x, Vector& y) {

	// This line and the next two lines should be removed and your version of ComputeSPMV should be used.
	A.isSpmvOptimized = true;

	assert(x.localLength >= A.localNumberOfColumns);
	assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif
	const double* const xv = x.values;
	double* const yv = y.values;
	const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	for (local_int_t i = 0; i < nrow; i++) {
		double sum = 0.0;
		for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
			local_int_t curCol = A.mtxIndL[i][j];
			sum += A.matrixValues[i][j] * xv[curCol];
		}
		yv[i] = sum;
	}

	return 0;
}
