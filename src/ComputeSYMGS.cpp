
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#include <cassert>
#include "ExchangeHalo.hpp"
#include <iostream>
#include "config.hpp"
#include "a_contral.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

int ComputeSYMGS_zcy(const SparseMatrix& A, const Vector& r, Vector& x);
int ComputeSYMGS_lmb(const SparseMatrix &A, const Vector &r, Vector &x);

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {
  if (g_optimization_type == OPTIM_TYPE_REF) {
#if defined(DebugPrintExecuteCalls)
    std::cout << "ComputeSYMGS_ref" << std::endl;
#endif
    return ComputeSYMGS_ref(A, r, x);
  }
  else if (g_optimization_type == OPTIM_TYPE_ZCY) {
#if defined(DebugPrintExecuteCalls)
    std::cout << "ComputeSYMGS_zcy" << std::endl;
#endif
    return ComputeSYMGS_zcy(A, r, x);
  }
  else {
#if defined(DebugPrintExecuteCalls)
    std::cout << "ComputeSYMGS_lmb" << std::endl;
#endif
    return ComputeSYMGS_lmb(A, r, x);
  }
}


int ComputeFusedSYMGS_SPMV_zcy(const SparseMatrix& A, const Vector& r, Vector& x, Vector& y) {
  assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const double* const rv = r.values;
  double* const xv = x.values;
  double* const yv = y.values;
  double** matrixDiagonal = A.matrixDiagonal;

  /*
   * FORWARD
   */
  for (local_int_t l = 0; l < A.tdg->size(); l++) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < (*A.tdg)[l].size(); i++) {
      local_int_t row = (*A.tdg)[l][i];
      const double* const currentValues = A.matrixValues[row];
      const local_int_t* const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0];
      double sum = rv[row];

      for (local_int_t j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[row] * currentDiagonal;
      xv[row] = sum / currentDiagonal;
    }
  }

  /*
   * BACKWARD (fusing SYMGS and SPMV)
   */
  for (local_int_t l = A.tdg->size() - 1; l >= 0; l--) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = (*A.tdg)[l].size() - 1; i >= 0; i--) {
      local_int_t row = (*A.tdg)[l][i];
      const double* const currentValues = A.matrixValues[row];
      const local_int_t* const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0];
      double sum = 0.0;

      for (local_int_t j = currentNumberOfNonzeros - 1; j >= 0; j--) {
        local_int_t curCol = currentColIndices[j];
        sum += currentValues[j] * xv[curCol];
      }
      sum -= xv[row] * currentDiagonal;
      xv[row] = (rv[row] - sum) / currentDiagonal;
      sum += xv[row] * currentDiagonal;
      yv[row] = sum;
    }
  }

  return 0;
}

int ComputeSYMGS_TDG_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {

  assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const double* const rv = r.values;
  double* const xv = x.values;
  double** matrixDiagonal = A.matrixDiagonal;

  /*
   * FORWARD
   */
  for (local_int_t l = 0; l < A.tdg->size(); l++) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < (*A.tdg)[l].size(); i++) {
      local_int_t row = (*A.tdg)[l][i];
      const double* const currentValues = A.matrixValues[row];
      const local_int_t* const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0];
      double sum = rv[row];

      for (local_int_t j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[row] * currentDiagonal;
      xv[row] = sum / currentDiagonal;
    }
  }

  /*
   * BACKWARD
   */
  for (local_int_t l = A.tdg->size() - 1; l >= 0; l--) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = (*A.tdg)[l].size() - 1; i >= 0; i--) {
      local_int_t row = (*A.tdg)[l][i];
      const double* const currentValues = A.matrixValues[row];
      const local_int_t* const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0];
      double sum = rv[row];

      for (local_int_t j = currentNumberOfNonzeros - 1; j >= 0; j--) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[row] * currentDiagonal;
      xv[row] = sum / currentDiagonal;
    }
  }

  return 0;
}

int ComputeSYMGS_BLOCK_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {

  assert(x.localLength >= A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double** matrixDiagonal = A.matrixDiagonal;
  const double* const rv = r.values;
  double* const xv = x.values;

  local_int_t firstBlock = 0;
  local_int_t lastBlock = firstBlock + (*A.numberOfBlocksInColor)[0];
  /*
   * FORWARD
   */
  for (local_int_t color = 0; color < A.numberOfColors; color++) {
    if (color > 0) {
      firstBlock += (*A.numberOfBlocksInColor)[color - 1];
      lastBlock = firstBlock + (*A.numberOfBlocksInColor)[color];
    }
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize) {
      local_int_t firstRow = block * A.blockSize;
      local_int_t firstChunk = firstRow / A.chunkSize;
      local_int_t lastChunk = (firstRow + A.blockSize * A.chunkSize) / A.chunkSize;

      for (local_int_t chunk = firstChunk; chunk < lastChunk; chunk++) {
        local_int_t first = A.chunkSize * chunk;
        local_int_t last = first + A.chunkSize;

        local_int_t i = first;
        if (A.chunkSize == 4) {
          double sum0 = rv[i + 0];
          double sum1 = rv[i + 1];
          double sum2 = rv[i + 2];
          double sum3 = rv[i + 3];

          for (local_int_t j = 0; j < (*A.nonzerosInChunk)[chunk]; j++) {
            sum0 -= A.matrixValues[i + 0][j] * xv[A.mtxIndL[i + 0][j]];
            sum1 -= A.matrixValues[i + 1][j] * xv[A.mtxIndL[i + 1][j]];
            sum2 -= A.matrixValues[i + 2][j] * xv[A.mtxIndL[i + 2][j]];
            sum3 -= A.matrixValues[i + 3][j] * xv[A.mtxIndL[i + 3][j]];
          }
          sum0 += matrixDiagonal[i + 0][0] * xv[i + 0];
          xv[i + 0] = sum0 / matrixDiagonal[i + 0][0];
          sum1 += matrixDiagonal[i + 1][1] * xv[i + 1];
          xv[i + 1] = sum1 / matrixDiagonal[i + 1][0];
          sum2 += matrixDiagonal[i + 2][2] * xv[i + 2];
          xv[i + 2] = sum2 / matrixDiagonal[i + 2][0];
          sum3 += matrixDiagonal[i + 3][3] * xv[i + 3];
          xv[i + 3] = sum3 / matrixDiagonal[i + 3][0];
        }
        else if (A.chunkSize == 2) {
          double sum0 = rv[i + 0];
          double sum1 = rv[i + 1];

          for (local_int_t j = 0; j < (*A.nonzerosInChunk)[chunk]; j++) {
            sum0 -= A.matrixValues[i + 0][j] * xv[A.mtxIndL[i + 0][j]];
            sum1 -= A.matrixValues[i + 1][j] * xv[A.mtxIndL[i + 1][j]];
          }
          sum0 += matrixDiagonal[i + 0][0] * xv[i + 0];
          xv[i + 0] = sum0 / matrixDiagonal[i + 0][0];
          sum1 += matrixDiagonal[i + 1][1] * xv[i + 1];
          xv[i + 1] = sum1 / matrixDiagonal[i + 1][0];
        }
        else if (A.chunkSize == 1) {
          double sum0 = rv[i + 0];

          for (local_int_t j = 0; j < (*A.nonzerosInChunk)[chunk]; j++) {
            sum0 -= A.matrixValues[i + 0][j] * xv[A.mtxIndL[i + 0][j]];
          }
          sum0 += matrixDiagonal[i + 0][0] * xv[i + 0];
          xv[i + 0] = sum0 / matrixDiagonal[i + 0][0];
        }
      }
    }
  }

  firstBlock = A.numberOfBlocks - 1;
  lastBlock = firstBlock - (*A.numberOfBlocksInColor)[A.numberOfColors - 1];
  /*
   * BACKWARD
   */
  for (local_int_t color = A.numberOfColors - 1; color >= 0; color--) {
    if (color < A.numberOfColors - 1) {
      firstBlock -= (*A.numberOfBlocksInColor)[color + 1];
      lastBlock = firstBlock - (*A.numberOfBlocksInColor)[color];
    }
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize) {
      local_int_t firstRow = ((block + 1) * A.blockSize) - 1; // this is the last row of the last block
      local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
      local_int_t lastChunk = (firstRow - A.blockSize * A.chunkSize) / A.chunkSize;

      for (local_int_t chunk = firstChunk; chunk > lastChunk; chunk--) {
        local_int_t first = A.chunkSize * chunk;
        local_int_t last = first + A.chunkSize;

        if (A.chunkSize == 4) {
          local_int_t i = last - 1;
          double sum3 = rv[i - 3];
          double sum2 = rv[i - 2];
          double sum1 = rv[i - 1];
          double sum0 = rv[i];

          for (local_int_t j = (*A.nonzerosInChunk)[chunk] - 1; j >= 0; j--) {
            sum3 -= A.matrixValues[i - 3][j] * xv[A.mtxIndL[i - 3][j]];
            sum2 -= A.matrixValues[i - 2][j] * xv[A.mtxIndL[i - 2][j]];
            sum1 -= A.matrixValues[i - 1][j] * xv[A.mtxIndL[i - 1][j]];
            sum0 -= A.matrixValues[i][j] * xv[A.mtxIndL[i][j]];
          }
          sum3 += matrixDiagonal[i - 3][0] * xv[i - 3];
          xv[i - 3] = sum3 / matrixDiagonal[i - 3][0];

          sum2 += matrixDiagonal[i - 2][1] * xv[i - 2];
          xv[i - 2] = sum2 / matrixDiagonal[i - 2][0];

          sum1 += matrixDiagonal[i - 1][2] * xv[i - 1];
          xv[i - 1] = sum1 / matrixDiagonal[i - 1][0];

          sum0 += matrixDiagonal[i][3] * xv[i];
          xv[i] = sum0 / matrixDiagonal[i][0];
        }
        else if (A.chunkSize == 2) {
          local_int_t i = last - 1;
          double sum1 = rv[i - 1];
          double sum0 = rv[i];

          for (local_int_t j = (*A.nonzerosInChunk)[chunk] - 1; j >= 0; j--) {
            sum1 -= A.matrixValues[i - 1][j] * xv[A.mtxIndL[i - 1][j]];
            sum0 -= A.matrixValues[i][j] * xv[A.mtxIndL[i][j]];
          }
          sum1 += matrixDiagonal[i - 1][2] * xv[i - 1];
          xv[i - 1] = sum1 / matrixDiagonal[i - 1][0];

          sum0 += matrixDiagonal[i][3] * xv[i];
          xv[i] = sum0 / matrixDiagonal[i][0];
        }
        else if (A.chunkSize == 1) {
          local_int_t i = last - 1;
          double sum0 = rv[i];

          for (local_int_t j = (*A.nonzerosInChunk)[chunk] - 1; j >= 0; j--) {
            sum0 -= A.matrixValues[i][j] * xv[A.mtxIndL[i][j]];
          }
          sum0 += matrixDiagonal[i][3] * xv[i];
          xv[i] = sum0 / matrixDiagonal[i][0];
        }
      }
    }
  }

  return 0;
}

int ComputeSYMGS_zcy(const SparseMatrix& A, const Vector& r, Vector& x) {

  // This function is just a stub right now which decides which implementation of the SYMGS will be executed (TDG or block coloring)
  if (A.TDG) {
    return ComputeSYMGS_TDG_zcy(A, r, x);
  }
  return ComputeSYMGS_BLOCK_zcy(A, r, x);
}

// A*x=r
// 对称高斯赛德尔迭代
int ComputeSYMGS_lmb(const SparseMatrix &A, const Vector &r, Vector &x)
{
	assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x); // 交换边界信息
#endif

#ifndef HPCG_USE_ELL // 不使用ELL格式
	const local_int_t nrow = A.localNumberOfRows;
	double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the diagonal entries A.matrixValues
	const double *const rv = r.values;
	double *const xv = x.values;

	for (local_int_t i = 0; i < nrow; i++)
	{
		const double *const currentValues = A.matrixValues[i];
		const local_int_t *const currentColIndices = A.mtxIndL[i];
		const int currentNumberOfNonzeros = A.nonzerosInRow[i];
		const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
		double sum = rv[i];									 // RHS value

		for (int j = 0; j < currentNumberOfNonzeros; j++)
		{
			local_int_t curCol = currentColIndices[j];
			sum -= currentValues[j] * xv[curCol];
		}
		sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

		xv[i] = sum / currentDiagonal;
	}

	// Now the back sweep.

	for (local_int_t i = nrow - 1; i >= 0; i--)
	{
		const double *const currentValues = A.matrixValues[i];
		const local_int_t *const currentColIndices = A.mtxIndL[i];
		const int currentNumberOfNonzeros = A.nonzerosInRow[i];
		const double currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
		double sum = rv[i];									 // RHS value

		for (int j = 0; j < currentNumberOfNonzeros; j++)
		{
			local_int_t curCol = currentColIndices[j];
			sum -= currentValues[j] * xv[curCol];
		}
		sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

		xv[i] = sum / currentDiagonal;
	}
#endif

#ifdef HPCG_USE_ELL // 使用ELL格式

#ifndef HPCG_USE_MULTICOLORING					  // 不使用染色法
	const local_int_t nrow = A.localNumberOfRows; // 此进程的行数

	// double **matrixDiagonal = A.matrixDiagonal;	  // 矩阵对角线项的指针 An array of pointers to the diagonal entries A.matrixValues
	double **matrixDiagonal = A.ellDiag;

	const double *const rv = r.values;
	double *const xv = x.values;

	double *ellVal = A.ellVal;
	// double **ellDiag = A.ellDiag;

	local_int_t *ellCols = &A.ellCols[0]; // 列索引

	for (local_int_t i = 0; i < nrow; i++) // 遍历所有结点
	{

		// 重写
		double sum = rv[i];
		const double currentDiagonal = matrixDiagonal[i][0];

		for (int j = 0; j < ELL_SIZE; j++)
		{

			if (ellCols[ELL_SIZE * i + j] == -1)
			{
				break;
			}
			sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
		}

		sum += xv[i] * currentDiagonal;
		xv[i] = sum / currentDiagonal;
	}

	for (local_int_t i = nrow - 1; i >= 0; i--)
	{
		// 重写
		double sum = rv[i];
		const double currentDiagonal = matrixDiagonal[i][0];

		for (int j = 0; j < ELL_SIZE; j++)
		{
			if (ellCols[ELL_SIZE * i + j] == -1)
				break;
			sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
		}
		sum += xv[i] * currentDiagonal;
		xv[i] = sum / currentDiagonal;
	}

#else // 使用染色法

#ifdef HPCG_USE_POINT_MULTICOLORING		 // 使用单点染色
#ifndef HPCG_USE_MULTICOLORING_RRARRANGE // 不使用重排

	//***************染色、不重排 ***************
	const local_int_t nrow = A.localNumberOfRows; // 此进程的行数

	double **matrixDiagonal = A.ellDiag; // 对角项

	const double *const rv = r.values;
	double *const xv = x.values;

	double *ellVal = A.ellVal;
	local_int_t *ellCols = &A.ellCols[0]; // 列索引

	int totalColors = A.totalColors;				  // 颜色总数
	const std::vector<local_int_t> colors = A.colors; // 染色标记

	for (int curColor = 0; curColor < totalColors; curColor++) // 遍历所有颜色,同一种颜色可以并行
	{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (local_int_t i = 0; i < nrow; i++)
		{
			if (colors[i] != curColor)
				continue;
			// 重写
			double sum = rv[i];
			const double currentDiagonal = matrixDiagonal[i][0];

			for (int j = 0; j < ELL_SIZE; j++)
			{

				if (ellCols[ELL_SIZE * i + j] == -1)
				{
					break;
				}
				sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
			}

			sum += xv[i] * currentDiagonal;
			xv[i] = sum / currentDiagonal;
		}
	}
	for (int curColor = totalColors - 1; curColor >= 0; curColor--)
	{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (local_int_t i = nrow - 1; i >= 0; i--)
		{
			if (colors[i] != curColor)
				continue;
			// 重写
			double sum = rv[i];
			const double currentDiagonal = matrixDiagonal[i][0];

			for (int j = 0; j < ELL_SIZE; j++)
			{
				if (ellCols[ELL_SIZE * i + j] == -1)
					break;
				sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
			}
			sum += xv[i] * currentDiagonal;
			xv[i] = sum / currentDiagonal;
		}
	}


#else // 使用重排

	const local_int_t nrow = A.localNumberOfRows; // 此进程的行数
	// double **matrixDiagonal = A.matrixDiagonal;	  // 矩阵对角线项的指针 An array of pointers to the diagonal entries A.matrixValues
	double **matrixDiagonal = A.ellDiag;
	const double *const rv = r.values;
	double *const xv = x.values;

	double *ellVal = A.ellVal;
	// double **ellDiag = A.ellDiag;
	local_int_t *ellCols = &A.ellCols[0]; // 列索引

	int totalColors = A.totalColors;				  // 颜色总数
	const std::vector<local_int_t> colors = A.colors; // 染色标记

	// 同一种颜色可以并行

	// #ifndef HPCG_NO_OPENMP
	// #pragma omp parallel num_threads(totalColors)
	// 	{
	// 		int thread_id = omp_get_thread_num();
	// #endif
	for (int curColor = 0; curColor < totalColors; curColor++) // 遍历所有color
	{
		local_int_t start = A.colorStart[curColor];
		local_int_t end = A.colorStart[curColor + 1];

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (int i = start; i < end; i++) // 遍历当前颜色节点
		{
			// 重写
			const double currentDiagonal = matrixDiagonal[i][0];
			local_int_t notRearrange = A.toNotRearrange[i]; // 获取未重排的索引
			double sum = rv[notRearrange];
			for (int j = 0; j < ELL_SIZE; j++)
			{

				if (ellCols[ELL_SIZE * i + j] == -1)
				{
					break;
				}
				sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
			}

			sum += xv[notRearrange] * currentDiagonal; // 去除对角元素项
			xv[notRearrange] = sum / currentDiagonal;
		}
	}

	// Now the back sweep.

	for (int curColor = totalColors - 1; curColor >= 0; curColor--)
	{
		local_int_t start = A.colorStart[curColor];
		local_int_t end = A.colorStart[curColor + 1];
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (local_int_t i = end - 1; i >= start; i--)
		{
			// 重写
			const double currentDiagonal = matrixDiagonal[i][0];
			local_int_t notRearrange = A.toNotRearrange[i]; // 获取重排前的索引
			double sum = rv[notRearrange];

			for (int j = 0; j < ELL_SIZE; j++)
			{
				if (ellCols[ELL_SIZE * i + j] == -1)
					break;
				sum -= ellVal[ELL_SIZE * i + j] * xv[ellCols[ELL_SIZE * i + j]];
			}
			sum += xv[notRearrange] * currentDiagonal;
			xv[notRearrange] = sum / currentDiagonal;
		}
	}



#endif
#endif

#ifdef HPCG_USE_BLOCK_MULTICOLORING // 使用分块染色
	//*************** 分块染色、不重排 ***************
	const local_int_t nrow = A.localNumberOfRows; // 此进程的行数

	double **matrixDiagonal = A.ellDiag; // 对角项

	const double *const rv = r.values;
	double *const xv = x.values;

	double *ellVal = A.ellVal;
	local_int_t *ellCols = &A.ellCols[0]; // 列索引

	int totalColors = A.totalColors;				  // 颜色总数
	const std::vector<local_int_t> colors = A.colors; // 染色标记

	int nx = A.geom->nx; // 子网格x方向尺寸

	for (int curColor = 0; curColor < totalColors; curColor++) // 遍历所有颜色,同一种颜色可以并行
	{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (local_int_t i = 0; i < nrow; i += nx) // 并行所有颜色相同的块
		{
			if (colors[i] != curColor)
				continue;

			for (local_int_t blockIdx = i; blockIdx < i+nx; blockIdx++) // 一个线程计算x方向分块
			{
				double sum = rv[blockIdx];
				const double currentDiagonal = matrixDiagonal[blockIdx][0];
				for (int j = 0; j < ELL_SIZE; j++)
				{

					if (ellCols[ELL_SIZE * blockIdx + j] == -1)
					{
						break;
					}
					sum -= ellVal[ELL_SIZE * blockIdx + j] * xv[ellCols[ELL_SIZE * blockIdx + j]];
				}
				sum += xv[blockIdx] * currentDiagonal;
				xv[blockIdx] = sum / currentDiagonal;
			}
		}
	}

	for (int curColor = totalColors-1; curColor >= 0; curColor--)
	{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for shared(colors, xv, matrixDiagonal)
#endif
		for (local_int_t i = nrow - nx; i >= 0; i -= nx) // 并行所有颜色相同的块
		{
			if (colors[i] != curColor)
				continue;

			for (local_int_t blockIdx = i + nx - 1; blockIdx >= i; blockIdx--)
			//for (local_int_t blockIdx = i; blockIdx < nx; blockIdx++)
			{
				double sum = rv[blockIdx];
				const double currentDiagonal = matrixDiagonal[blockIdx][0];
				for (int j = 0; j < ELL_SIZE; j++)
				{
					if (ellCols[ELL_SIZE * blockIdx + j] == -1)
						break;
					sum -= ellVal[ELL_SIZE * blockIdx + j] * xv[ellCols[ELL_SIZE * blockIdx + j]];
				}
				sum += xv[blockIdx] * currentDiagonal;
				xv[blockIdx] = sum / currentDiagonal;
			}
		}
	}

#endif

#endif

#endif
	return 0;

	// This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
	// return ComputeSYMGS_ref(A, r, x);
}
