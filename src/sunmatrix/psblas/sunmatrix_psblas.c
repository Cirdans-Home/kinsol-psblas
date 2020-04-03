/*
 * -----------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * Based on code sundials_sparse.c by: Carol Woodward and
 *     Slaven Peles @ LLNL, and Daniel R. Reynolds @ SMU
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the implementation file for the PSBLAS implementation of
 * the SUNMATRIX package.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_psblas.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_math.h>

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/*
 * ==================================================================
 * Private function prototypes (functions working on SlsMat)
 * ==================================================================
 */

/* ----------------------------------------------------------------------------
 * Function to create a new sparse matrix
 */

SUNMatrix SUNPSBLASMatrix(int ictxt, psb_c_descriptor *cdh)
{
  SUNMatrix A;
  SUNMatrix_Ops ops;
  SUNMatrixContent_PSBLAS content;

  return(A);
}



/* ----------------------------------------------------------------------------
 * Function to create a new sparse matrix from an existing dense matrix
 * by copying all nonzero values into the sparse matrix structure.  Returns NULL
 * if the request for matrix storage cannot be satisfied.
 */

SUNMatrix SUNPSBLASFromDenseMatrix(SUNMatrix Ad, realtype droptol, int sparsetype, int ictxt, psb_c_descriptor *cdh)
{
  SUNMatrix As;

  return(As);
}


/* ----------------------------------------------------------------------------
 * Function to create a new sparse matrix from an existing band matrix
 * by copying all nonzero values into the sparse matrix structure.  Returns NULL
 * if the request for matrix storage cannot be satisfied.
 */

SUNMatrix SUNPSBLASFromBandMatrix(SUNMatrix Ad, realtype droptol, int sparsetype, int ictxt, psb_c_descriptor *cdh)
{
  SUNMatrix As;

  return(As);
}

/* ----------------------------------------------------------------------------
 * Function to print the sparse matrix
 */

void SUNPSBLASMatrix_Print(SUNMatrix A, FILE* outfile)
{

  return;
}


/* ----------------------------------------------------------------------------
 * Functions to access the contents of the sparse matrix structure
 */

sunindextype SUNPSBLASMatrix_Rows(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
    return 0;
  else
    return -1;
}

sunindextype SUNPSBLASMatrix_Columns(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
    return 0;
  else
    return -1;
}

sunindextype SUNPSBLASMatrix_NNZ(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
    return 0;
  else
    return -1;
}

int SUNPSBLASMatrix_PSBLASType(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
    return 0;
  else
    return -1;
}

realtype* SUNPSBLASMatrix_Data(SUNMatrix A)
{
  realtype dummy[10];

  if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
    return dummy;
  else
    return NULL;
}

/*
 * -----------------------------------------------------------------
 * implementation of matrix operations
 * -----------------------------------------------------------------
 */

SUNMatrix_ID SUNMatGetID_PSBLAS(SUNMatrix A)
{
  return SUNMATRIX_CUSTOM;
}

SUNMatrix SUNMatClone_PSBLAS(SUNMatrix A)
{
  SUNMatrix B;

  return(B);
}

void SUNMatDestroy_PSBLAS(SUNMatrix A)
{
  return;
}

int SUNMatZero_PSBLAS(SUNMatrix A)
{
  return 0;
}

int SUNMatCopy_PSBLAS(SUNMatrix A, SUNMatrix B)
{

  return 0;
}

int SUNMatScaleAddI_PSBLAS(realtype c, SUNMatrix A)
{

  return 0;
}

int SUNMatScaleAdd_PSBLAS(realtype c, SUNMatrix A, SUNMatrix B)
{
  /* return success */
  return(0);

}

int SUNMatMatvec_PSBLAS(SUNMatrix A, N_Vector x, N_Vector y)
{

  return 0;
}
