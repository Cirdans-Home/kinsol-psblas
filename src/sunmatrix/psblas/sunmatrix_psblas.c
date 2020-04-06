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
#include <nvector/nvector_psblas.h>
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
  psb_c_dspmat *ah;
  psb_i_t ret;

  /* Create Matrix */
  A = NULL;
  A = (SUNMatrix) malloc(sizeof *A);
  if (A == NULL) return(NULL);

  /* Create matrix operation structure */
  ops = NULL;
  ops = (SUNMatrix_Ops) malloc(sizeof(struct _generic_SUNMatrix_Ops));
  if (ops == NULL) { free(A); return(NULL); }

  /* Attach operations */
  ops->getid       = SUNMatGetID_PSBLAS;
  ops->clone       = SUNMatClone_PSBLAS;
  ops->destroy     = SUNMatDestroy_PSBLAS;
  ops->zero        = SUNMatZero_PSBLAS;
  ops->copy        = SUNMatCopy_PSBLAS;
  ops->scaleadd    = SUNMatScaleAdd_PSBLAS;
  ops->scaleaddi   = SUNMatScaleAddI_PSBLAS;
  ops->matvec      = SUNMatMatvec_PSBLAS;
  ops->space       = SUNMatSpace_PSBLAS;

  /* Create content */
  content = NULL;
  content = (SUNMatrixContent_PSBLAS) malloc(sizeof(struct _SUNMatrixContent_Sparse));
  if (content == NULL) { free(ops); free(A); return(NULL); }

  content->ictxt = ictxt;
  content->cdh   = cdh;
  ah  = psb_c_new_dspmat();
  ret = psb_c_dspall(ah,cdh);
  if(ret == 0){
    free(content);
    free(ops);
    free(A);
    return(NULL);
  }
  content->ah = ah;

  /* Attach content and ops */
  A->content = content;
  A->ops     = ops;

  return(A);
}



/* ----------------------------------------------------------------------------
 * Function to create a new sparse matrix from an existing dense matrix
 * by copying all nonzero values into the sparse matrix structure.  Returns NULL
 * if the request for matrix storage cannot be satisfied.
 */

SUNMatrix SUNPSBLASFromDenseMatrix(SUNMatrix A, realtype droptol, int ictxt, psb_c_descriptor *cdh)
{
  SUNMatrix As;

  return(As);
}


/* ----------------------------------------------------------------------------
 * Function to create a new sparse matrix from an existing band matrix
 * by copying all nonzero values into the sparse matrix structure.  Returns NULL
 * if the request for matrix storage cannot be satisfied.
 */

SUNMatrix SUNPSBLASFromBandMatrix(SUNMatrix A, realtype droptol, int ictxt, psb_c_descriptor *cdh)
{
  sunindextype i, j, nnz;
  sunindextype M, N;
  SUNMatrix As;



  return(As);
}

/* ----------------------------------------------------------------------------
 * Function to create a new PSBLAS matrix from an existing sparse matrix
 * by copying all nonzero values into the PSBLAS matrix structure.  Returns NULL
 * if the request for matrix storage cannot be satisfied.
 */

SUNMatrix SUNPSBLASFromSparseMatrix(SUNMatrix A, int ictxt, psb_c_descriptor *cdh)
{
  SUNMatrix As;

  return(As);
}

/* ----------------------------------------------------------------------------
 * Function to print the sparse matrix
 */

void SUNPSBLASMatrix_Print(SUNMatrix A, char *matrixtitle, char* filename)
{

  psb_c_dmm_mat_write( SM_PMAT_P(A), matrixtitle, filename);

  return;
}


/* ----------------------------------------------------------------------------
 * Functions to access the contents of the sparse matrix structure
 */

sunindextype SUNPSBLASMatrix_Rows(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_CUSTOM)
    return(psb_c_dmat_get_nrows(SM_PMAT_P(A)));
  else
    return -1;
}

sunindextype SUNPSBLASMatrix_Columns(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_CUSTOM)
    return(psb_c_dmat_get_ncols(SM_PMAT_P(A)));
  else
    return -1;
}

sunindextype SUNPSBLASMatrix_NNZ(SUNMatrix A)
{
  if (SUNMatGetID(A) == SUNMATRIX_CUSTOM)
    return(psb_c_dnnz(SM_PMAT_P(A),SM_DESCRIPTOR_P(A)));
  else
    return -1;
}

realtype* SUNPSBLASMatrix_Data(SUNMatrix A)
{
  realtype dummy[10];

  if (SUNMatGetID(A) == SUNMATRIX_CUSTOM)
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
  SUNMatrix_Ops ops;
  SUNMatrixContent_PSBLAS content;
  psb_c_dspmat *ah;
  psb_i_t ret;

  /* Create Matrix */
  B = NULL;
  B = (SUNMatrix) malloc(sizeof *A);
  if (B == NULL) return(NULL);

  /* Create matrix operation structure */
  ops = NULL;
  ops = (SUNMatrix_Ops) malloc(sizeof(struct _generic_SUNMatrix_Ops));
  if (ops == NULL) { free(B); return(NULL); }

  /* Attach operations */
  ops->getid       = SUNMatGetID_PSBLAS;
  ops->clone       = SUNMatClone_PSBLAS;
  ops->destroy     = SUNMatDestroy_PSBLAS;
  ops->zero        = SUNMatZero_PSBLAS;
  ops->copy        = SUNMatCopy_PSBLAS;
  ops->scaleadd    = SUNMatScaleAdd_PSBLAS;
  ops->scaleaddi   = SUNMatScaleAddI_PSBLAS;
  ops->matvec      = SUNMatMatvec_PSBLAS;
  ops->space       = SUNMatSpace_PSBLAS;

  /* Create content */
  content = NULL;
  content = (SUNMatrixContent_PSBLAS) malloc(sizeof(struct _SUNMatrixContent_Sparse));
  if (content == NULL) { free(ops); free(A); return(NULL); }

  content->ictxt = SM_ICTXT_P(A);
  content->cdh   = SM_DESCRIPTOR_P(A);
  ah  = psb_c_new_dspmat();
  ret = psb_c_dspall(ah,SM_DESCRIPTOR_P(A));
  if(ret == 0){
    free(content);
    free(ops);
    free(B);
    return(NULL);
  }
  content->ah = ah;

  /* Attach content and ops */
  B->content = content;
  B->ops     = ops;

  return(B);
}

void SUNMatDestroy_PSBLAS(SUNMatrix A)
{
  /* Free the PSBLAS Sparse Matrix */
  psb_c_dspfree(SM_PMAT_P(A), SM_DESCRIPTOR_P(A));
  /* Free the NMATRIX_PSBLAS Structure */
  free(A->content); A->content = NULL;
  free(A->ops);  A->ops = NULL;
  free(A); A = NULL;

  return;
}

int SUNMatZero_PSBLAS(SUNMatrix A)
{
  bool clear = SUNTRUE;
  // The PSBLAS matrix contained in A will be in the UPDATE state at the of
  // this call
  psb_c_dsprn(SM_PMAT_P(A), SM_DESCRIPTOR_P(A), clear);
  return 0;
}

int SUNMatCopy_PSBLAS(SUNMatrix A, SUNMatrix B)
{
  return(psb_c_dcopy_mat(SM_PMAT_P(A),SM_PMAT_P(B),SM_DESCRIPTOR_P(A)));
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

  psb_c_dspmm( (psb_d_t) 1.0, SM_PMAT_P(A), NV_PVEC_P(x) ,
  		    (psb_d_t) 1.0, NV_PVEC_P(y) , NV_DESCRIPTOR_P(x));

  return 0;
}
