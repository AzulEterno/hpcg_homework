
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include "Vector.hpp"
#include <iostream>


#include <cstring>
#include "a_contral.hpp"

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif


int OptimizeProblem_lmb(SparseMatrix& A, CGData& data, Vector& b, Vector& x, Vector& xexact);

int OptimizeCoarseProblem_zcy(SparseMatrix& A) {

  const local_int_t nrow = A.localNumberOfRows;

  // On the coarse grids we use block coloring algorithm
  A.TDG = false; //Multi-Level dependency graph indicater 

  local_int_t nyPerBlock = 1; // i.e., how many whole-NX is a block
  A.blockSize = A.geom->nx * nyPerBlock;

  // How many blocks we found in each direction?
  local_int_t blocksInX = 1; // at least, one block has NX rows
  local_int_t blocksInY = A.geom->ny / nyPerBlock;
  local_int_t blocksInZ = A.geom->nz; // blocks cannot contain rows from different X-Y planes

  local_int_t numberOfBlocks = nrow / A.blockSize;
  A.numberOfBlocks = numberOfBlocks;

  // Our target is to have two colors per each X-Y plane
  // If not possible, then we will repeat colors at some point
  local_int_t targetNumberOfColors = (2 * A.geom->nz);
  if (A.numberOfBlocks % targetNumberOfColors != 0) {
    targetNumberOfColors = A.geom->nz;
  }

  local_int_t blocksPerColor = A.numberOfBlocks / targetNumberOfColors;

  // Find  a good chunkSize
  if (blocksPerColor % 4 == 0) {
    A.chunkSize = 4;
  }
  else if (blocksPerColor % 2 == 0) {
    A.chunkSize = 2;
  }
  else {
    A.chunkSize = 1;
  }

  assert(A.numberOfBlocks % A.chunkSize == 0);
  assert(targetNumberOfColors % 2 == 0);

  A.firstRowOfBlock = new std::vector<local_int_t>(numberOfBlocks);
  for (local_int_t i = 0, ii = 0; i < nrow; i += A.blockSize, ii++) {
    (*A.firstRowOfBlock)[ii] = i;
  }

  // Create an adjacency matrix for the blocked grid
  local_int_t** blockIndices = new local_int_t * [numberOfBlocks];
  for (local_int_t i = 0; i < numberOfBlocks; i++) {
    blockIndices[i] = new local_int_t[27];
  }
  local_int_t* nonzerosInBlock = new local_int_t[numberOfBlocks];
  for (local_int_t z = 0; z < blocksInZ; z++) {
    for (local_int_t y = 0; y < blocksInY; y++) {
      for (local_int_t x = 0; x < blocksInX; x++) {
        local_int_t curBlock = x + y * blocksInX + z * blocksInX * blocksInY;
        local_int_t* ptr = blockIndices[curBlock];
        local_int_t nnzInBlock = 0;

        for (int zz = -1; zz <= 1; zz++) {
          if (z + zz >= 0 && z + zz < blocksInZ) {
            for (int yy = -1; yy <= 1; yy++) {
              if (y + yy >= 0 && y + yy < blocksInY) {
                for (int xx = -1; xx <= 1; xx++) {
                  if (x + xx >= 0 && x + xx < blocksInX) {
                    local_int_t colBlock = curBlock + xx + yy * blocksInX + zz * blocksInX * blocksInY;
                    *ptr++ = colBlock;
                    nnzInBlock++;
                  }
                }
              }
            }
          }
        }
        nonzerosInBlock[curBlock] = nnzInBlock;
      }
    }
  }

  // We can start coloring
  std::vector<local_int_t> colors(numberOfBlocks, numberOfBlocks); // `numberOfBlocks` means uninitialized
  local_int_t totalColors = 1;
  colors[0] = 0; // first block gets color 0

  for (local_int_t i = 1; i < numberOfBlocks; i++) {
    if (colors[i] == numberOfBlocks) { // if color is not assigned to this block
      std::vector<local_int_t> assigned(totalColors, 0);
      local_int_t currentlyAssigned = 0;
      const local_int_t* const currentColIndices = blockIndices[i];
      const int nnz = nonzerosInBlock[i];
      for (local_int_t j = 0; j < nnz; j++) {
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // points beyond i are unassigned, don't about before i
          if (assigned[colors[curCol]] == 0) {
            currentlyAssigned++;
          }
          assigned[colors[curCol]] = 1;
        }
        else {
          break; // indices sorted, we can break
        }
      }
      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (local_int_t j = 0; j < totalColors; j++) {
          if (assigned[j] == 0) { // no neighbour block has this color
            colors[i] = j;
            break;
          }
        }
      }
      else { // all colors assigned, create a new one and assign it
        colors[i] = totalColors++;
      }
    }
  }

  // Increment the number of colors by changing the colors of some rows
  if (totalColors < targetNumberOfColors) {
    local_int_t colorIncrement = targetNumberOfColors - totalColors;
    for (local_int_t i = 0; i < numberOfBlocks; i += 2 * blocksInX * blocksInY) {
      colorIncrement = colorIncrement >= (targetNumberOfColors - totalColors) ? 0 : colorIncrement + 4;
      for (local_int_t ii = i; ii < i + blocksInX * blocksInY; ii++) {
        colors[ii] += colorIncrement;
      }

    }
    totalColors = targetNumberOfColors;
  }
  A.numberOfColors = totalColors;

  std::vector<std::vector<local_int_t> > blocksInColor(totalColors);
  for (local_int_t i = 0; i < numberOfBlocks; i++) {
    blocksInColor[colors[i]].push_back(i);
  }

  A.numberOfBlocksInColor = new std::vector<local_int_t>(totalColors);
  for (local_int_t c = 0; c < totalColors; c++) {
    (*A.numberOfBlocksInColor)[c] = blocksInColor[c].size();
  }

  // Allocate memory for temporary data structures
  double** matrixValues = new double* [nrow];
  local_int_t** mtxIndL = new local_int_t * [nrow];
  unsigned char* nonzerosInRow = new unsigned char[nrow];
  for (local_int_t i = 0; i < nrow; i++) {
    matrixValues[i] = new double[27];
    mtxIndL[i] = new local_int_t[27];
  }

  // Populate new data structures. Also, merge blocks now
  // We will create the translation vectors on the fly as well
  A.whichOldRowIsNewRow = new std::vector<local_int_t>(A.localNumberOfColumns);
  A.whichNewRowIsOldRow = new std::vector<local_int_t>(A.localNumberOfColumns);
  local_int_t ptr = 0;
  for (local_int_t color = 0; color < totalColors; color++) {
    for (local_int_t block = 0; block < blocksInColor[color].size(); block += A.chunkSize) {
      for (local_int_t i = 0; i < A.blockSize; i++) {
        for (local_int_t b = 0; b < A.chunkSize; b++) {
          local_int_t curBlock = blocksInColor[color][block + b];
          local_int_t firstRow = (*A.firstRowOfBlock)[curBlock];
          local_int_t curRow = firstRow + i;

          for (local_int_t j = 0; j < A.nonzerosInRow[curRow]; j++) {
            matrixValues[ptr][j] = A.matrixValues[curRow][j];
            mtxIndL[ptr][j] = A.mtxIndL[curRow][j];
          }
          nonzerosInRow[ptr] = A.nonzerosInRow[curRow];
          (*A.whichOldRowIsNewRow)[ptr] = curRow;
          (*A.whichNewRowIsOldRow)[curRow] = ptr++;
        }
      }
    }
  }
  // External rows are not reordered so they keep the same ID
  for (local_int_t i = nrow; i < A.localNumberOfColumns; i++) {
    (*A.whichOldRowIsNewRow)[i] = i;
    (*A.whichNewRowIsOldRow)[i] = i;
  }

  // Scan the grid to discover the amount of nonzeros per chunk
  // We already consider the new order
  A.numberOfChunks = nrow / A.chunkSize;
  A.nonzerosInChunk = new std::vector<local_int_t>(A.numberOfChunks, 0);
  for (local_int_t i = 0; i < nrow; i += A.chunkSize) {
    local_int_t curChunk = i / A.chunkSize;
    (*A.nonzerosInChunk)[curChunk] = nonzerosInRow[i];
    for (local_int_t ii = i + 1; ii < i + A.chunkSize; ii++) {
      (*A.nonzerosInChunk)[curChunk] = (*A.nonzerosInChunk)[curChunk] < nonzerosInRow[ii] ? nonzerosInRow[ii] : (*A.nonzerosInChunk)[curChunk];
    }
  }

  // Translate indices
  for (local_int_t c = 0; c < A.numberOfChunks; c++) {
    for (local_int_t i = 0; i < A.chunkSize; i++) {
      local_int_t curRow = c * A.chunkSize + i;
      for (local_int_t j = 0; j < nonzerosInRow[curRow]; j++) {
        local_int_t curCol = mtxIndL[curRow][j];
        mtxIndL[curRow][j] = (*A.whichNewRowIsOldRow)[curCol];
      }
    }
  }

  // Make sure the values from nonzerosInRow->nonzerosInChunk are actually 0
  for (local_int_t i = 0; i < nrow; i++) {
    local_int_t curChunk = i / A.chunkSize;

    for (local_int_t j = nonzerosInRow[i]; j < (*A.nonzerosInChunk)[curChunk]; j++) {
      matrixValues[i][j] = 0.0;
      mtxIndL[i][j] = 0;
    }
  }

  // Regenerate the firstRowOfBlock data structure
  ptr = 0;
  for (local_int_t c = 0; c < totalColors; c++) {
    for (local_int_t i = 0; i < blocksInColor[c].size(); i++) {
      (*A.firstRowOfBlock)[blocksInColor[c][i]] = ptr;
      ptr += A.blockSize;
    }
  }

  // Replace data structures
  for (local_int_t i = 0; i < nrow; i++) {
    A.nonzerosInRow[i] = nonzerosInRow[i];
    for (local_int_t j = 0; j < 27; j++) {
      A.matrixValues[i][j] = matrixValues[i][j];
      A.mtxIndL[i][j] = mtxIndL[i][j];
    }
  }

  // Regenerate the diagonal
  for (local_int_t i = 0; i < nrow; i++) {
    local_int_t curChunk = i / A.chunkSize;
    for (local_int_t j = 0; j < (*A.nonzerosInChunk)[curChunk]; j++) {
      local_int_t curCol = A.mtxIndL[i][j];
      if (i == curCol) {
        A.matrixDiagonal[i] = &A.matrixValues[i][j];
        break;
      }
    }
  }

#ifndef HPCG_NO_MPI
  // Translate row IDs that will be send to neighbours
  for (local_int_t i = 0; i < A.totalToBeSent; i++) {
    local_int_t orig = A.elementsToSend[i];
    A.elementsToSend[i] = (*A.whichNewRowIsOldRow)[orig];
  }
#endif

#ifdef HPCG_USE_SPMV_ARMPL
  // Create the temporary data structures that will be copied (flags=0) inside the create call
  local_int_t m = A.localNumberOfRows;
  local_int_t n = A.localNumberOfColumns;

  local_int_t nnz = A.localNumberOfNonzeros;

  armpl_int_t* row_ptr = (armpl_int_t*)std::malloc((m + 1) * sizeof(armpl_int_t));
  armpl_int_t* col_idx = (armpl_int_t*)std::malloc(nnz * sizeof(armpl_int_t));
  double* vals = (double*)std::malloc(nnz * sizeof(double));

  row_ptr[0] = 0;
  for (local_int_t i = 0; i < m; i++) {
    row_ptr[i + 1] = row_ptr[i] + A.nonzerosInRow[i];
  }

  global_int_t k = 0;
  for (local_int_t i = 0; i < m; i++) {
    for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
      col_idx[k] = A.mtxIndL[i][j];
      vals[k++] = A.matrixValues[i][j];
    }
  }

  armpl_int_t flags = 0;
  armpl_spmat_create_csr_d(&A.armpl_mat, m, n, row_ptr, col_idx, vals, flags);

  free(row_ptr);
  free(col_idx);
  free(vals);

  armpl_spmat_hint(A.armpl_mat, ARMPL_SPARSE_HINT_STRUCTURE, ARMPL_SPARSE_STRUCTURE_HPCG);
  armpl_spmv_optimize(A.armpl_mat);
#endif

  //Clean heap memory part
  if (blockIndices) {
    for (local_int_t i = 0; i < numberOfBlocks; i++) {
      delete[] blockIndices[i];
    }
    delete[] blockIndices;
  }

  if (nonzerosInBlock) {
    delete[] nonzerosInBlock;
  }

  if (matrixValues) {

    for (local_int_t i = 0; i < nrow; i++) {
      delete[] matrixValues[i];
    }
    delete[] matrixValues;

  }

  if (mtxIndL) {
    for (local_int_t i = 0; i < nrow; i++) {
      delete[] mtxIndL[i];
    }
    delete[] mtxIndL;
  }

  if (nonzerosInRow) {
    delete[] nonzerosInRow;
  }

  //End of clean Heal memory


  if (A.mgData != 0) {
    // Translate f2cOperator
    local_int_t ncrow = (A.geom->nx / 2) * (A.geom->ny / 2) * (A.geom->nz / 2);
    for (local_int_t i = 0; i < ncrow; i++) {
      local_int_t orig = A.mgData->f2cOperator[i];
      A.mgData->f2cOperator[i] = (*A.whichNewRowIsOldRow)[orig];
    }

    return OptimizeCoarseProblem_zcy(*A.Ac);
  }




  return 0;

}



/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
  */
int OptimizeProblemGeneral_zcy(SparseMatrix& A, CGData& data, Vector& b, Vector& x, Vector& xexact) {

  const local_int_t nrow = A.localNumberOfRows;

  // On the finest grid we use TDG algorithm
  A.TDG = true;

  // Create an auxiliary vector to store the number of dependencies on L for every row
  std::vector<unsigned char> nonzerosInLowerDiagonal(nrow, 0);

  /*
   * Now populate these vectors. This loop is safe to parallelize
   */
#if defined(ENABLE_DEBUG_PRINT)
  printf("Optimize problem Phase 1 Started\n");
#endif
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nrow; i++) {
    for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
      local_int_t curCol = A.mtxIndL[i][j];
      if (curCol < i && curCol < nrow) { // check that it's on L and not a row from other domain
        nonzerosInLowerDiagonal[i]++;
      }
      else if (curCol == i) { // we found the diagonal, no more L dependencies from here
        break;
      }
    }
  }
#if defined(ENABLE_DEBUG_PRINT)
  printf("Optimize problem Phase 2 Started\n");
#endif
  std::vector<local_int_t> depsVisited(nrow, 0);
  std::vector<bool> processed(nrow, false);
  local_int_t rowsProcessed = 0;

  //printf("Local variable initalized \n");

  //if (A.tdg)
    // Allocate the TDG structure. Starts as an empty matrix 

  A.tdg = new std::vector<std::vector<local_int_t> >();
  //printf("A.tdg initalized \n");
  // We start by adding the first row of the grid to the first level. This row has no L dependencies
  std::vector<local_int_t> aux(1, 0);
  A.tdg->push_back(aux);
  // Increment the number of dependencies visited for each of the neighbours
  for (local_int_t j = 0; j < A.nonzerosInRow[0]; j++) {
    //printf("j = %d \n", j);
    if (A.mtxIndL[0][j] != 0 && A.mtxIndL[0][j] < nrow) {
      depsVisited[A.mtxIndL[0][j]]++; // don't update deps from other domains
    }
  }
  processed[0] = true;
  rowsProcessed++;

#ifdef ENABLE_DEBUG_PRINT
  printf("Optimize problem Phase 3 Started\n");
#endif
  // Continue with the creation of the TDG
  while (rowsProcessed < nrow) {
    std::vector<local_int_t> rowsInLevel;

    // Check for the dependencies of the rows of the level before the current one. The dependencies
    // of these rows are the ones that could have their dependencies fulfilled and therefore added to the
    // current level
    unsigned int lastLevelOfTDG = A.tdg->size() - 1;
    for (local_int_t i = 0; i < (*A.tdg)[lastLevelOfTDG].size(); i++) {
      local_int_t row = (*A.tdg)[lastLevelOfTDG][i];

      for (local_int_t j = 0; j < A.nonzerosInRow[row]; j++) {
        local_int_t curCol = A.mtxIndL[row][j];

        if (curCol < nrow) { // don't process external domain rows
          // If this neighbour hasn't been processed yet and all its L dependencies has been processed
          if (!processed[curCol] && depsVisited[curCol] == nonzerosInLowerDiagonal[curCol]) {
            rowsInLevel.push_back(curCol); // add the row to the new level
            processed[curCol] = true; // mark the row as processed
          }
        }
      }
    }




    // Update some information
    for (local_int_t i = 0; i < rowsInLevel.size(); i++) {
      rowsProcessed++;
      local_int_t row = rowsInLevel[i];
      for (local_int_t j = 0; j < A.nonzerosInRow[row]; j++) {
        local_int_t curCol = A.mtxIndL[row][j];
        if (curCol < nrow && curCol != row) {
          depsVisited[curCol]++;
        }
      }
    }

    // Add the just created level to the TDG structure
    A.tdg->push_back(rowsInLevel);
  }

#if defined(ENABLE_DEBUG_PRINT)
  printf("Optimize problem Phase 4 Started\n");
#endif
  // Now we need to create some structures to translate from old and new order (yes, we will reorder the matrix)
  A.whichNewRowIsOldRow = new std::vector<local_int_t>(A.localNumberOfColumns);
  A.whichOldRowIsNewRow = new std::vector<local_int_t>(A.localNumberOfColumns);

  local_int_t oldRow = 0;
  for (local_int_t level = 0; level < A.tdg->size(); level++) {
    for (local_int_t i = 0; i < (*A.tdg)[level].size(); i++) {
      local_int_t newRow = (*A.tdg)[level][i];
      (*A.whichOldRowIsNewRow)[oldRow] = newRow;
      (*A.whichNewRowIsOldRow)[newRow] = oldRow++;
    }
  }

  // External domain rows are not reordered, thus they keep the same ID
  for (local_int_t i = nrow; i < A.localNumberOfColumns; i++) {
    (*A.whichOldRowIsNewRow)[i] = i;
    (*A.whichNewRowIsOldRow)[i] = i;
  }


  // Now we need to allocate some structure to temporary allocate the reordered structures
  double** matrixValues = new double* [nrow];
  local_int_t** mtxIndL = new local_int_t * [nrow];
  char* nonzerosInRow = new char[nrow];

  for (local_int_t i = 0; i < nrow; i++) {
    matrixValues[i] = new double[27];
    mtxIndL[i] = new local_int_t[27];
  }

  // And finally we reorder (and translate at the same time)
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t level = 0; level < A.tdg->size(); level++) {
    for (local_int_t i = 0; i < (*A.tdg)[level].size(); i++) {
      local_int_t oldRow = (*A.tdg)[level][i];
      local_int_t newRow = (*A.whichNewRowIsOldRow)[oldRow];

      nonzerosInRow[newRow] = A.nonzerosInRow[oldRow];
      for (local_int_t j = 0; j < A.nonzerosInRow[oldRow]; j++) {
        local_int_t curOldCol = A.mtxIndL[oldRow][j];
        matrixValues[newRow][j] = A.matrixValues[oldRow][j];
        mtxIndL[newRow][j] = curOldCol < nrow ? (*A.whichNewRowIsOldRow)[curOldCol] : curOldCol; // don't translate if row is external
      }
    }
  }

  // time to replace structures
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nrow; i++) {
    A.nonzerosInRow[i] = nonzerosInRow[i];
    for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
      A.matrixValues[i][j] = matrixValues[i][j];
      A.mtxIndL[i][j] = mtxIndL[i][j];
    }
    // Put some zeros on padding positions
    for (local_int_t j = A.nonzerosInRow[i]; j < 27; j++) {
      A.matrixValues[i][j] = 0.0;
      A.mtxIndL[i][j] = 0;
    }
  }




#if defined(ENABLE_DEBUG_PRINT)
  printf("Optimize problem Phase 5 Started\n");
#endif
  // Regenerate the diagonal
  for (local_int_t i = 0; i < nrow; i++) {
    for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
      local_int_t curCol = A.mtxIndL[i][j];
      if (i == curCol) {
        A.matrixDiagonal[i] = &A.matrixValues[i][j];
      }
    }
  }

  // Translate TDG row IDs
  oldRow = 0;
  for (local_int_t l = 0; l < A.tdg->size(); l++) {
    for (local_int_t i = 0; i < (*A.tdg)[l].size(); i++) {
      (*A.tdg)[l][i] = oldRow++;
    }
  }

#ifndef HPCG_NO_MPI
  // Translate the row IDs that will be send to other domains
  for (local_int_t i = 0; i < A.totalToBeSent; i++) {
    local_int_t orig = A.elementsToSend[i];
    A.elementsToSend[i] = (*A.whichNewRowIsOldRow)[orig];
  }
#endif

  // Reorder b (RHS) vector
  Vector bReorder;
  InitializeVector(bReorder, b.localLength);
  CopyVector(b, bReorder);
  CopyAndReorderVector(bReorder, b, *A.whichNewRowIsOldRow);

#ifdef HPCG_USE_SPMV_ARMPL
  // Create the temporary data structures that will be copied (flags=0) inside the create call
  local_int_t m = A.localNumberOfRows;
  local_int_t n = A.localNumberOfColumns;

  local_int_t nnz = A.localNumberOfNonzeros;

  armpl_int_t* row_ptr = (armpl_int_t*)std::malloc((m + 1) * sizeof(armpl_int_t));
  armpl_int_t* col_idx = (armpl_int_t*)std::malloc(nnz * sizeof(armpl_int_t));
  double* vals = (double*)std::malloc(nnz * sizeof(double));

  row_ptr[0] = 0;
  for (local_int_t i = 0; i < m; i++) {
    row_ptr[i + 1] = row_ptr[i] + A.nonzerosInRow[i];
  }

  global_int_t k = 0;
  for (local_int_t i = 0; i < m; i++) {
    for (local_int_t j = 0; j < A.nonzerosInRow[i]; j++) {
      col_idx[k] = A.mtxIndL[i][j];
      vals[k++] = A.matrixValues[i][j];
    }
  }

  armpl_int_t flags = 0;
  armpl_spmat_create_csr_d(&A.armpl_mat, m, n, row_ptr, col_idx, vals, flags);

  free(row_ptr);
  free(col_idx);
  free(vals);

  armpl_spmat_hint(A.armpl_mat, ARMPL_SPARSE_HINT_STRUCTURE, ARMPL_SPARSE_STRUCTURE_HPCG);
  armpl_spmv_optimize(A.armpl_mat);
#endif

  //Clean heap memory
  if (matrixValues) {

    for (local_int_t i = 0; i < nrow; i++) {
      delete[] matrixValues[i];
    }

    delete[] matrixValues;
  }

  if (mtxIndL) {

    for (local_int_t i = 0; i < nrow; i++) {
      delete[] mtxIndL[i];
    }

    delete[] mtxIndL;
  }
  if (nonzerosInRow) {
    delete[] nonzerosInRow;
  }

  DeleteVector(bReorder);
  //End of clean heap memory

  if (A.mgData != 0) {
#if defined(ENABLE_DEBUG_PRINT)
    printf("Optimize problem Phase 6 Started\n");
#endif
    // Translate f2cOperator values
    local_int_t ncrow = (A.geom->nx / 2) * (A.geom->ny / 2) * (A.geom->nz / 2);
    for (local_int_t i = 0; i < ncrow; i++) {
      local_int_t orig = A.mgData->f2cOperator[i];
      A.mgData->f2cOperator[i] = (*A.whichNewRowIsOldRow)[orig];
    }
#if defined(ENABLE_DEBUG_PRINT)
    printf("Optimize problem Phase 6 Ended\n");
#endif
    return OptimizeCoarseProblem_zcy(*A.Ac);
  }

  return 0;
}

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix& A, CGData& data, Vector& b, Vector& x, Vector& xexact) {
  if (g_optimization_type == OPTIM_TYPE_REF) {
    // Do nothing
#if defined(DebugPrintExecuteCalls)
    std::cout << "OptimizeProblem do nothing" << std::endl;
#endif
    return 0;
  }
  else if (g_optimization_type == OPTIM_TYPE_ZCY) {
#if defined(DebugPrintExecuteCalls)
    std::cout << "OptimizeProblemGeneral_zcy" << std::endl;
#endif
    return OptimizeProblemGeneral_zcy(A, data, b, x, xexact);
  }
  else {
#if defined(DebugPrintExecuteCalls)
    std::cout << "OptimizeProblem_lmb" << std::endl;
#endif
    return OptimizeProblem_lmb(A, data, b, x, xexact);
  }
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix& A) {

  return 0.0;

}

// 转换至ELL格式
void OptimizeMatrixToEll(SparseMatrix& A)
{
  const int nrows = A.localNumberOfRows;
  // int idx;
  // MPI_Comm_rank(MPI_COMM_WORLD, &idx);
  // if (idx == 0)
  // {
  // 	printf("%d\n", nrows);
  // 	//printf("%lld\n", A.totalNumberOfRows);
  // 	//printf("%d\n", ELL_SIZE * nrows);
  // }

  // 内存分配
  A.ellVal = new double[ELL_SIZE * nrows];
  A.ellCols = new local_int_t[ELL_SIZE * nrows];
  A.ellDiag = new double* [nrows];

  // allocateMemoryToOptimaize+=

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for // 将CSR格式转为ELL存储
#endif
  for (int i = 0; i < nrows; i++)
  {
    const double* const cur_vals = A.matrixValues[i]; // 第i个矩阵
    const local_int_t* const cur_idx = A.mtxIndL[i];  // 列索引
    const int cur_nnz = A.nonzerosInRow[i];			  // 非零元素的数量

    int non_zero_idx = 0;
    // printf("ELL-%d:",i);
    for (int j = 0; j < cur_nnz; j++) // 遍历非零元素
    {
      // printf("%f ", A.mtxIndL[i][j]);
      A.ellVal[i * ELL_SIZE + j] = cur_vals[j];
      A.ellCols[i * ELL_SIZE + j] = cur_idx[j];
      // 更新对角线索引矩阵

      if (A.matrixDiagonal[i] == &cur_vals[j]) // 判断是否为对角线元素
      {
        // printf("对角线元素：%f\n",*A.matrixDiagonal[i]);
        // A.matrixDiagonal[i] = &A.ellVal[i * ELL_SIZE + j]; // 更新对角线元素索引
        A.ellDiag[i] = &A.ellVal[i * ELL_SIZE + j]; // 更新对角线元素索引
      }

      non_zero_idx++;
    }
    // std::cout<<A.matrixDiagonal[i]<<" : "<<A.ellDiag[i]<<std::endl;
    //  printf("\n");
    for (; non_zero_idx < ELL_SIZE; non_zero_idx++)
    {
      A.ellCols[i * ELL_SIZE + non_zero_idx] = -1; // 空位置置为-1
    }
  }
}

// 单点染色
void OptimizeMultiColoring(SparseMatrix& A)
{
  const local_int_t nrow = A.localNumberOfRows;

  // 初始化所有颜色为'nrow'，表示未初始化
  // std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  A.colors = std::vector<local_int_t>(nrow, nrow);
  A.totalColors = 1;
  A.colors[0] = 0; // 为第一个点分配颜色0 first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i = 1; i < nrow; ++i)
  {
    if (A.colors[i] == nrow) // 如果颜色未分配
    {						 // if color not assigned
      std::vector<int> assigned(A.totalColors, 0);
      int currentlyAssigned = 0; // 当强已经被分配颜色

      local_int_t* currentColIndices = &A.ellCols[i * ELL_SIZE]; // 列索引
      // const local_int_t *const currentColIndices = A.mtxIndL[i];

      // const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j = 0; j < ELL_SIZE; j++) // 如果颜色未分配
      {
        if (currentColIndices[j] == -1)
          break;
        local_int_t curCol = currentColIndices[j]; // 遍历邻居, 邻近点列索引

        if (curCol < i) // 邻居点已经被染色
        {
          if (assigned[A.colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[A.colors[curCol]] = 1; // 这个颜色已被'curCol'点使用 this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < A.totalColors)		// 如果至少有一种颜色可用
      {											// if there is at least one color left to use
        for (int j = 0; j < A.totalColors; ++j) // try all current colors
          if (assigned[j] == 0)
          { // if no neighbor with this color
            A.colors[i] = j;
            break;
          }
      }
      else // 没有颜色可用
      {
        if (A.colors[i] == nrow)
        {
          A.colors[i] = A.totalColors;
          A.totalColors += 1;
        }
      }
    }
  }
  return;
}

// 分块染色
void OptimizeMultiColoringBlock(SparseMatrix& A)
{
  const local_int_t nrow = A.localNumberOfRows;

  A.colors = std::vector<local_int_t>(nrow, nrow);
  A.totalColors = 1;
  A.colors[0] = 0; // 为第一个点分配颜色0 first point gets color 0

  int nx = A.geom->nx; // 子网格的尺寸
  int npy = A.geom->ny;
  int npz = A.geom->nz;


  for (local_int_t i = nx; i < nrow; i += nx) // 以npx为间隔
  {
    if (A.colors[i] == nrow) // 如果颜色未分配
    {
      std::vector<int> assigned(A.totalColors, 0); // 初始化邻居被染色标记数组
      int currentlyAssigned = 0;					 // 当前已经被分配颜色

      local_int_t* currentColIndices = &A.ellCols[i * ELL_SIZE]; // 列索引

      for (int j = 0; j < ELL_SIZE; j++) // 遍历邻居结点
      {
        if (currentColIndices[j] == -1)
          break;
        local_int_t curCol = currentColIndices[j]; // 邻近点列索引
        if (curCol % nx == 0)  // yz平面的块头结点
        {
          if (curCol < i) // 邻居点已经被染色
          {
            if (assigned[A.colors[curCol]] == 0)
              currentlyAssigned += 1;
            assigned[A.colors[curCol]] = 1; // 这个颜色已被'curCol'点使用 this color has been used before by `curCol' point
          } // else // could take advantage of indices being sorted
        }
      }

      if (currentlyAssigned < A.totalColors)		// 如果至少有一种颜色可用
      {											// if there is at least one color left to use
        for (int j = 0; j < A.totalColors; ++j) // try all current colors
          if (assigned[j] == 0)
          { // if no neighbor with this color
            A.colors[i] = j;
            break;
          }
      }
      else // 没有颜色可用
      {
        if (A.colors[i] == nrow)
        {
          A.colors[i] = A.totalColors;
          A.totalColors += 1;
        }
      }
    }
  }
  //printf("染色总数%d\n",A.totalColors);

  return;
}

// // 染色粗矩阵
// void OptimizeMultiColoringCoarse(SparseMatrix &A)
// {
// 	SparseMatrix *coarseMatrix = A.Ac;				  // 粗网格数据指针
// 	const local_int_t nrow = A.Ac->localNumberOfRows; // 粗网格的行数
// 	local_int_t *f2cOperator = A.mgData->f2cOperator; // 粗网格--->细网格的行索引
//
// 	coarseMatrix->colors = std::vector<local_int_t>(nrow, nrow);
// 	coarseMatrix->totalColors = A.totalColors;
//
// 	for (local_int_t i = 0; i < nrow; i++)
// 	{
// 		local_int_t fineRow = f2cOperator[i]; // 粗网格对应细网格的行索引
// 		coarseMatrix->colors[i] = A.colors[fineRow];
// 	}
// }

// 染色重排
void OptimizeMultiColoringRearrange(SparseMatrix& A)
{
  // 内存分配

  // double *tempB = new double[b.localLength]; // 向量b重排的buffer

  const local_int_t nrows = A.localNumberOfRows;

  A.toNotRearrange = new local_int_t[nrows];

  double* tempVal = new double[ELL_SIZE * nrows];
  double** tempDiag = new double* [nrows];
  local_int_t* tempCol = new local_int_t[ELL_SIZE * nrows];

  int totalColors = A.totalColors;
  std::vector<local_int_t> colors = A.colors;
  local_int_t* colorStart = new local_int_t[totalColors + 1]; // 记录每种颜色的开始索引

  colorStart[totalColors] = nrows; // 索引截至

  int cur = 0;
  for (int i = 0; i < totalColors; i++) // 遍历颜色
  {
    colorStart[i] = cur;
    for (int j = 0; j < nrows; j++)
    {
      if (colors[j] == i) // 节点颜色与当前相同
      {
        A.toNotRearrange[cur] = j;
        // 矩阵内存拷贝
        memcpy(&tempVal[cur * ELL_SIZE], &A.ellVal[j * ELL_SIZE], ELL_SIZE * sizeof(double));
        // 列索引拷贝
        memcpy(&tempCol[cur * ELL_SIZE], &A.ellCols[j * ELL_SIZE], ELL_SIZE * sizeof(local_int_t));
        // if (rank == 0)
        // {
        // 	tempB[cur] = b.values[j];
        // }

        for (int k = 0; k < ELL_SIZE; k++)
        {
          // tempVal[cur * ELL_SIZE + k] = A.ellVal[j * ELL_SIZE + k];
          // tempCol[cur * ELL_SIZE + k] = A.ellCols[j * ELL_SIZE + k];

          if (&A.ellVal[j * ELL_SIZE + k] == A.ellDiag[j]) // 判断是否为对角线元素
          {
            tempDiag[cur] = &tempVal[cur * ELL_SIZE + k];
          }
        }

        // tempDiag[cur] = A.matrixDiagonal[j];
        cur++; // 游标自加
      }
    }
  }

  // // 释放未重排的数组
  delete[] A.ellCols;
  delete[] A.ellVal;
  delete[] A.ellDiag;
  // delete[] b.values;
  //  delete[] A.matrixDiagonal;

  A.ellVal = tempVal;
  A.ellCols = tempCol;
  A.ellDiag = tempDiag;
  // A.matrixDiagonal = tempDiag;
  A.colorStart = colorStart;
  // b.values = tempB;
  printf("染色重排结束\n");
}

// 问题优化
int OptimizeProblem_lmb(SparseMatrix& A, CGData& data, Vector& b, Vector& x, Vector& xexact)
{
  int idx;
  MPI_Comm_rank(MPI_COMM_WORLD, &idx); // 获取进程id

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

#if defined(HPCG_USE_ELL)
  if (idx == 0)
    printf("转变为ELL存储\n");
  // 从csr格式变为ELL格式
  OptimizeMatrixToEll(A);

  // 将粗网格转变为ELL格式
  int numberOfMgLevels = 4; // 粗网格的层次
  SparseMatrix* coreseMatrix = &A;
  for (int level = 1; level < numberOfMgLevels; level++)
  {
    coreseMatrix = coreseMatrix->Ac;
    OptimizeMatrixToEll(*coreseMatrix);
  }
#endif

  // 染色法预处理
#if defined(HPCG_USE_POINT_MULTICOLORING)
  if (idx == 0)
    printf("单点染色\n");
  OptimizeMultiColoring(A); // 将矩阵A染色

  // 粗处理网格染色
  coreseMatrix = &A;
  for (int level = 1; level < numberOfMgLevels; level++)
  {
    //OptimizeMultiColoringCoarse(*coreseMatrix);
    OptimizeMultiColoring(*coreseMatrix);
    coreseMatrix = coreseMatrix->Ac;
  }

#endif

  // 染色重排

#if defined(HPCG_USE_MULTICOLORING_RRARRANGE)
  if (idx == 0)
    printf("染色重排\n");
  OptimizeMultiColoringRearrange(A);
  SparseMatrix* curMatrix = &A; // 当前矩阵
  for (int level = 1; level < numberOfMgLevels; level++)
  {
    curMatrix = curMatrix->Ac;
    OptimizeMultiColoringRearrange(*curMatrix);
  }
#endif


#if defined(HPCG_USE_BLOCK_MULTICOLORING)
  if (idx == 0)
    printf("分块染色\n");
  OptimizeMultiColoringBlock(A);
  SparseMatrix* curMatrix = &A;  //当前矩阵

  for (int level = 1; level < numberOfMgLevels; level++)
  {
    curMatrix = curMatrix->Ac;
    OptimizeMultiColoringBlock(*curMatrix);
  }
#endif
  return 0;
}
