/*
 * -----------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * Based on code sundials_sparse.h by: Carol Woodward and
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
 * This is the header file for the psblas implementation of the
 * SUNMATRIX module, SUNMATRIX_PSBLAS.
 *
 * Notes:
 *   - The definition of the generic SUNMatrix structure can be found
 *     in the header file sundials_matrix.h.
 *   - The definition of the type 'realtype' can be found in the
 *     header file sundials_types.h, and it may be changed (at the
 *     configuration stage) according to the user's needs.
 *     The sundials_types.h file also contains the definition
 *     for the type 'booleantype' and 'indextype'.
 * -----------------------------------------------------------------
 */

#ifndef _SUNMATRIX_PSBLAS_H
#define _SUNMATRIX_PSBLAS_H

#include <stdio.h>
#include <mpi.h>
#include "psb_base_cbind.h"
#include "psb_c_base.h"
#include "psb_util_cbind.h"
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_sparse.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* ------------------------
 * Matrix Type Definitions
 * ------------------------ */

#define CSC_MAT 0
#define CSR_MAT 1
#define PSBLAS_MAT 2


/* ------------------------------------------
 * Sparse Implementation of SUNMATRIX_SPARSE
 * ------------------------------------------ */

struct _SUNMatrixContent_PSBLAS {
  int sparsetype;
  psb_c_descriptor *cdh;       /* descriptor for data distribution */
  psb_c_dspmat  *ah;           /* PSBLAS sparse matrix             */
  int ictxt;                   /* PSBLAS communicator              */
};

typedef struct _SUNMatrixContent_PSBLAS *SUNMatrixContent_PSBLAS;


/* ---------------------------------------
 * Macros for access to SUNMATRIX_SPARSE
 * --------------------------------------- */

#define SM_CONTENT_P(A)     ( (SUNMatrixContent_PSBLAS)(A->content) )

#define SM_SPARSETYPE_P(A)  ( SM_CONTENT_P(A)->sparsetype )

#define SM_DESCRIPTOR_P(A)  ( SM_CONTENT_P(A)->cdh )

#define SM_PMAT_P(A)        ( SM_CONTENT_P(A)->ah )

#define SM_ICTXT_P(A)       ( SM_CONTENT_P(A)->ictxt )

/* ----------------------------------------
 * Exported Functions for SUNMATRIX_PSBLAS
 * ---------------------------------------- */

SUNDIALS_EXPORT SUNMatrix SUNPSBLASMatrix(int ictxt, psb_c_descriptor *cdh);

SUNDIALS_EXPORT SUNMatrix SUNPSBLASFromDenseMatrix(SUNMatrix A,
                                                   realtype droptol,
                                                   int sparsetype,
                                                   int ictxt,
                                                   psb_c_descriptor *cdh);

SUNDIALS_EXPORT SUNMatrix SUNPSBLASFromBandMatrix(SUNMatrix A,
                                                  realtype droptol,
                                                  int sparsetype,
                                                  int ictxt,
                                                  psb_c_descriptor *cdh);

SUNDIALS_EXPORT SUNMatrix SUNPSBLASFromSparseMatrix(SUNMatrix A,
                                                  int ictxt,
                                                  psb_c_descriptor *cdh);

// SUNDIALS_EXPORT int SUNPSBLASMatrix_Realloc(SUNMatrix A);

// SUNDIALS_EXPORT int SUNPSBLASMatrix_Reallocate(SUNMatrix A, sunindextype NNZ);

SUNDIALS_EXPORT void SUNPSBLASMatrix_Print(SUNMatrix A, char *matrixtitle, char* filename);

SUNDIALS_EXPORT sunindextype SUNPSBLASMatrix_Rows(SUNMatrix A);
SUNDIALS_EXPORT sunindextype SUNPSBLASMatrix_Columns(SUNMatrix A);
SUNDIALS_EXPORT sunindextype SUNPSBLASMatrix_NNZ(SUNMatrix A);
// SUNDIALS_EXPORT sunindextype SUNPSBLASMatrix_NP(SUNMatrix A);
SUNDIALS_EXPORT int SUNPSBLASMatrix_PSBLASType(SUNMatrix A);
SUNDIALS_EXPORT realtype* SUNPSBLASMatrix_Data(SUNMatrix A);
// SUNDIALS_EXPORT sunindextype* SUNPSBLASMatrix_IndexValues(SUNMatrix A);
// SUNDIALS_EXPORT sunindextype* SUNPSBLASMatrix_IndexPointers(SUNMatrix A);

SUNDIALS_EXPORT SUNMatrix_ID SUNMatGetID_PSBLAS(SUNMatrix A);
SUNDIALS_EXPORT SUNMatrix SUNMatClone_PSBLAS(SUNMatrix A);
SUNDIALS_EXPORT void SUNMatDestroy_PSBLAS(SUNMatrix A);
SUNDIALS_EXPORT int SUNMatZero_PSBLAS(SUNMatrix A);
SUNDIALS_EXPORT int SUNMatCopy_PSBLAS(SUNMatrix A, SUNMatrix B);
SUNDIALS_EXPORT int SUNMatScaleAdd_PSBLAS(realtype c, SUNMatrix A, SUNMatrix B);
SUNDIALS_EXPORT int SUNMatScaleAddI_PSBLAS(realtype c, SUNMatrix A);
SUNDIALS_EXPORT int SUNMatMatvec_PSBLAS(SUNMatrix A, N_Vector x, N_Vector y);
SUNDIALS_EXPORT int SUNMatSpace_PSBLAS(SUNMatrix A, long int *lenrw, long int *leniw);


#ifdef __cplusplus
}
#endif

#endif
