
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#if __cplusplus < 201103L
// for C++03
#include <map>
typedef std::map< global_int_t, local_int_t > GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
#endif
#include <iostream>
#include "config.hpp"
#include "a_contral.hpp"

#define ELL_SIZE 27

struct SparseMatrix_STRUCT {
  char  * title; //!< name of the sparse matrix
  Geometry * geom; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
  local_int_t localNumberOfRows; //!< number of rows local to this process
  local_int_t localNumberOfColumns;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
  char  * nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG; //!< matrix indices as global values
  local_int_t ** mtxIndL; //!< matrix indices as local values
  double ** matrixValues; //!< values of matrix entries
  double ** matrixDiagonal; //!< values of matrix diagonal entries
  GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable struct SparseMatrix_STRUCT * Ac; // Coarse grid matrix
  mutable MGData * mgData; // Pointer to the coarse level data for this fine matrix
  void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external to this process
  int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent; //!< total number of entries to be sent
  local_int_t * elementsToSend; //!< elements to send to neighboring processes
  int * neighbors; //!< neighboring processes
  local_int_t * receiveLength; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength; //!< lenghts of messages sent to neighboring processes
  double * sendBuffer; //!< send buffer for non-blocking sends
#endif

  // ZCY
  std::vector<local_int_t>* whichNewRowIsOldRow = nullptr;
  std::vector<local_int_t>* whichOldRowIsNewRow = nullptr;
  std::vector<local_int_t>* firstRowOfBlock = nullptr;
  std::vector<local_int_t>* nonzerosInChunk = nullptr;
  std::vector<std::vector<local_int_t>>* tdg = nullptr;
  std::vector<local_int_t>* numberOfBlocksInColor = nullptr;

  bool TDG;
  local_int_t blockSize;
  local_int_t chunkSize;
  local_int_t numberOfChunks;
  local_int_t numberOfColors;
  local_int_t numberOfBlocks;


  SparseMatrix_STRUCT() {


  }
  // ZCY

  // LMB
  /*!
	 This is for storing optimized data structres created in OptimizeProblem and
	 used inside optimized ComputeSPMV().
	 这用于存储在OptimizeProblem中创建的优化数据结构，并在优化的ComputeSPMV()中使用。
	 */

	// ELL格式存储
	double *ellVal;		  // ell数据
	double **ellDiag;	  // ell对角线元素
	local_int_t *ellCols; // ell列索引

	// 染色记录
	std::vector<local_int_t> colors; // 染色vector
	int totalColors;				 // 颜色总数
	local_int_t *toNotRearrange;	 // 重排索引到为重排索引

	// 染色重排
	local_int_t *colorStart; // 重排后颜色开始位置索引
  // LMB
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

  // LMB
  // 优化数据结构
	A.ellVal = NULL;
	A.ellDiag = NULL;
	A.ellCols = NULL;

	//A.colors = NULL;
	A.totalColors = 0;
	A.toNotRearrange = NULL;

	A.colorStart = NULL;
  // LMB

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

inline void CopyMatrixDiagonal_lmb(SparseMatrix &A, Vector &diagonal)
{
#ifndef HPCG_USE_ELL // 不使用ELL
	double **curDiagA = A.matrixDiagonal;
#else
	double **curDiagA = A.ellDiag;
#endif
	double *dv = diagonal.values;
	assert(A.localNumberOfRows == diagonal.localLength);
	for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
	{
		dv[i] = *(curDiagA[i]);
	}
	return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
  if (g_optimization_type == OPTIM_TYPE_LMB) {
    CopyMatrixDiagonal_lmb(A, diagonal);
  }
  else {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
  }
  return;
}

// 置换对角线元素
inline void ReplaceMatrixDiagonal_lmb(SparseMatrix &A, Vector &diagonal)
{
#ifndef HPCG_USE_ELL // 不使用ELL
	double **curDiagA = A.matrixDiagonal;
#else
	double **curDiagA = A.ellDiag;
#endif
	double *dv = diagonal.values;
	assert(A.localNumberOfRows == diagonal.localLength);
	for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
	{
		*(curDiagA[i]) = dv[i];
	}
	return;
}

/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
  if (g_optimization_type == OPTIM_TYPE_LMB) {
    ReplaceMatrixDiagonal_lmb(A, diagonal);
  }
  else {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];
  }
  return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix & A) {

#ifndef HPCG_CONTIGUOUS_ARRAYS
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }
#else
  delete [] A.matrixValues[0];
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndL[0];
#endif
  if (A.title)                  delete [] A.title;
  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
  if (A.mtxIndG) delete [] A.mtxIndG;
  if (A.mtxIndL) delete [] A.mtxIndL;
  if (A.matrixValues) delete [] A.matrixValues;
  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
  if (A.elementsToSend)       delete [] A.elementsToSend;
  if (A.neighbors)              delete [] A.neighbors;
  if (A.receiveLength)            delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  // ZCY
  //Clean support data
  if (true) {
    if (A.whichNewRowIsOldRow != nullptr) {
      delete A.whichNewRowIsOldRow;
      A.whichNewRowIsOldRow = nullptr;
    }

    if (A.whichOldRowIsNewRow != nullptr) {
      delete A.whichOldRowIsNewRow;
      A.whichOldRowIsNewRow = nullptr;
    }

    if (A.firstRowOfBlock != nullptr) {
      delete A.firstRowOfBlock;
      A.firstRowOfBlock = nullptr;
    }

    if (A.nonzerosInChunk != nullptr) {

      delete A.nonzerosInChunk;
      A.nonzerosInChunk = nullptr;
    }

    if (A.numberOfBlocksInColor != nullptr) {
      delete A.numberOfBlocksInColor;
      A.numberOfBlocksInColor = nullptr;
    }
    if (A.tdg != nullptr) {
      //A.tdg->clear();
      try {
        delete A.tdg;
        A.tdg = nullptr;
      }
      catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
      }
    }
    //Done with that
  }
  // ZCY

  // LMB
  if (A.ellVal != NULL)
		delete[] A.ellVal;
	if (A.ellDiag != NULL)
		delete[] A.ellDiag;
	if (A.ellCols != NULL)
		delete[] A.ellCols;
	if (A.toNotRearrange != NULL)
		delete[] A.toNotRearrange;
	if (A.colorStart)
		delete[] A.colorStart;
  // LMB

  if (A.geom!=0) { DeleteGeometry(*A.geom); delete A.geom; A.geom = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
  if (A.mgData!=0) { DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data
  return;
}

#endif // SPARSEMATRIX_HPP
