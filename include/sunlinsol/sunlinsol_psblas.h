/*
 * -----------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
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
 * This is the header file for the PSBLAS implementation of the
 * SUNLINSOL module, SUNLINSOL_PSBLAS.  This enables to use all the
 * Krylov algorithms in PSBLAS and the preconditioner implemented (or
 * accessed) in MLD2P4. See the manual for the options.
 *
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can
 *     be found in the header file sundials_linearsolver.h.
 * -----------------------------------------------------------------
 */

#ifndef _SUNLINSOL_PSBLAS_H
#define _SUNLINSOL_PSBLAS_H

#include <stdio.h>
#include <mpi.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>

#include "psb_base_cbind.h"
#include "psb_c_dbase.h"
#include "psb_util_cbind.h"
#include "psb_prec_cbind.h"
#include "psb_krylov_cbind.h"
#include "mld_c_dprec.h"
#include "mld_const.h"

#undef I

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* --------------------------------------
 * PSBLAS Implementation of SUNLinearSolver
 * -------------------------------------- */

struct _SUNLinearSolverContent_PSBLAS {
  psb_c_SolverOptions options; /* PSBLAS solver options                     */
  psb_c_dprec *ph;             /* PSBLAS preconditioner                     */
  mld_c_dprec *mh;             /* MLD2P4 preconditioner                     */
  psb_c_descriptor *cdh;       /* descriptor for data distribution          */
  psb_c_dspmat  *ah;           /* PSBLAS sparse matrix (Coefficient)        */
  psb_c_dspmat  *bh;           /* PSBLAS sparse matrix (Preconditioner)     */
  int ictxt;                   /* PSBLAS communicator                       */
  char methd[40];              /* String for Method and Preconditioner type */
  char ptype[20];
};

typedef struct _SUNLinearSolverContent_PSBLAS *SUNLinearSolverContent_PSBLAS;

/* ---------------------------------------
 * Macros for access to SUNLinSol_PSBLAS
 * --------------------------------------- */

#define PSBLAS_CONTENT(S)  ( (SUNLinearSolverContent_PSBLAS)(S->content) )

#define LS_PREC_P(S)  ( PSBLAS_CONTENT(S)->ph )

#define LS_MLPREC_P(S)  ( PSBLAS_CONTENT(S)->mh )

#define LS_DESCRIPTOR_P(S)  ( PSBLAS_CONTENT(S)->cdh )

#define LS_PMAT_P(S)  ( PSBLAS_CONTENT(S)->ah )

#define LS_BMAT_P(S)  ( PSBLAS_CONTENT(S)->bh )

#define LS_ICTXT_P(S) ( PSBLAS_CONTENT(S)->ictxt )

#define LS_METHD_P(S) ( PSBLAS_CONTENT(S)->methd )

#define LS_PTYPE_P(S) ( PSBLAS_CONTENT(S)->ptype )


/* ---------------------------------------
 * Exported Functions for SUNLINSOL_PSBLAS
 * --------------------------------------- */
SUNDIALS_EXPORT SUNLinearSolver SUNLinSol_PSBLAS(psb_c_SolverOptions options, char methd[], char ptype[], psb_i_t ictxt);
/* Core Functions */
SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolInitialize_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetup_PSBLAS(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_PSBLAS(SUNLinearSolver S, SUNMatrix A,
                                             N_Vector x, N_Vector b,
                                             realtype tol);
SUNDIALS_EXPORT int SUNLinSolFree_PSBLAS(SUNLinearSolver S);

/* Set functions */
SUNDIALS_EXPORT int SUNLinSolSeti_PSBLAS(SUNLinearSolver S, const char *what, psb_i_t val);
SUNDIALS_EXPORT int SUNLinSolSetc_PSBLAS(SUNLinearSolver S, const char *what, const char *val);
SUNDIALS_EXPORT int SUNLinSolSetr_PSBLAS(SUNLinearSolver S, const char *what, double val);
SUNDIALS_EXPORT int SUNLinSolSetPreconditioner_PSBLAS(SUNLinearSolver S, SUNMatrix B);
/* Get functions */
SUNDIALS_EXPORT int SUNLinSolNumIters_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT realtype SUNLinSolResNorm_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT long int SUNLinSolLastFlag_PSBLAS(SUNLinearSolver S);

#ifdef __cplusplus
}
#endif

#endif
