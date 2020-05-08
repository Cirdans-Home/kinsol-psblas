/*
 * -----------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * Based on sundials_spgmr.c code, written by Scott D. Cohen,
 *                Alan C. Hindmarsh and Radu Serban @ LLNL
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
 * This is the implementation file for the SPGMR implementation of
 * the SUNLINSOL package.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sunlinsol/sunlinsol_psblas.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_nvector.h>
#include <sunmatrix/sunmatrix_psblas.h>
#include <nvector/nvector_psblas.h>

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif


/* Function used to create a PSBLAS linear solver */
SUNLinearSolver SUNLinSol_PSBLAS(psb_c_SolverOptions options, char methd[], char ptype[], psb_i_t ictxt){

  SUNLinearSolver S;
  SUNLinearSolver_Ops ops;
  SUNLinearSolverContent_PSBLAS content;

  /* Create linear solver */
  S = NULL;
  S = (SUNLinearSolver) malloc(sizeof *S);
  if (S == NULL) return(NULL);

  /* Create linear solver operation structure */
  ops = NULL;
  ops = (SUNLinearSolver_Ops) malloc(sizeof(struct _generic_SUNLinearSolver_Ops));
  if (ops == NULL) { free(S); return(NULL); }

  /* Attach operations */
  ops->gettype           = SUNLinSolGetType_PSBLAS;
  ops->setatimes         = NULL;
  ops->setpreconditioner = NULL;
  ops->setscalingvectors = NULL;
  ops->initialize        = SUNLinSolInitialize_PSBLAS;
  ops->setup             = SUNLinSolSetup_PSBLAS;
  ops->solve             = SUNLinSolSolve_PSBLAS;
  ops->numiters          = SUNLinSolNumIters_PSBLAS;
  ops->resnorm           = SUNLinSolResNorm_PSBLAS;
  ops->resid             = NULL;
  ops->lastflag          = SUNLinSolLastFlag_PSBLAS;
  ops->space             = NULL;
  ops->free              = SUNLinSolFree_PSBLAS;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_PSBLAS) malloc(sizeof(struct _SUNLinearSolverContent_PSBLAS));
  if (content == NULL) { free(ops); free(S); return(NULL); }

  /* Fill content:
  Only the solver options are set at this stage, all the information regarding
  the communicator, the PSBLAS contex are imported from the matrix when the
  solver is initialized. The setup of the preconditioner has to be done with
  the other routine, here we decide only if we are using a PSBLAS or an MLD2P4
  preconditioner from the type.                                               */
  content->options=options;
  content->cdh=NULL;
  content->ah=NULL;
  content->ictxt=ictxt;
  strcpy(content->methd,methd);
  strcpy(content->ptype,ptype);
  /* We need to decide here between a PSBLAS or an MLD2P4 preconditioner, this
  depends on the ptype string */
  if( strcmp(ptype,"NONE")==0 || strcmp(ptype,"BJAC")==0 || strcmp(ptype,"DIAG")==0 ){
      content->ph=psb_c_new_dprec();
      content->mh=NULL;
  }else if( strcmp(ptype,"ML")==0 ){
      content->ph=NULL;
      content->mh=mld_c_dprec_new();
  }else{
    free(content);
    free(ops);
    free(S);
    return(NULL);
  }

  /* Attach content and ops */
  S->content = content;
  S->ops     = ops;

  return(S);
}

/* The four required routines to define a linear solver */

SUNLinearSolver_Type SUNLinSolGetType_PSBLAS(SUNLinearSolver S)
{
  return(SUNLINEARSOLVER_MATRIX_ITERATIVE);
}

int SUNLinSolInitialize_PSBLAS(SUNLinearSolver S){

  psb_i_t ret;
  psb_i_t iam,np;

  psb_c_info(LS_ICTXT_P(S),&iam,&np);

  if(iam==0) printf("I'm initializing the %s solver with %s preconditioner\n",LS_METHD_P(S),LS_PTYPE_P(S));

  if(S == NULL) return(SUNLS_MEM_NULL);

  if( strcmp(LS_PTYPE_P(S),"NONE")==0 ||
      strcmp(LS_PTYPE_P(S),"BJAC")==0 ||
      strcmp(LS_PTYPE_P(S),"DIAG")==0 ){
      printf("\n\nPSBLAS init!\n\n");
      ret = psb_c_dprecinit(LS_ICTXT_P(S),LS_PREC_P(S),LS_PTYPE_P(S));
      if(ret != 0){
        if(iam == 0) printf("Failure on PSBLAS precinit %d ictxt %d ptype %s\n",ret,LS_ICTXT_P(S),LS_PTYPE_P(S));
        return(SUNLS_PSET_FAIL_UNREC);
      }
  }else if(strcmp(LS_PTYPE_P(S),"ML") == 0 ||
    strcmp(LS_PTYPE_P(S),"GS") == 0 ||
    strcmp(LS_PTYPE_P(S),"AS") == 0 ||
    strcmp(LS_PTYPE_P(S),"FBGS") == 0 ){
      ret = mld_c_dprecinit(LS_ICTXT_P(S), LS_MLPREC_P(S), LS_PTYPE_P(S));
      if(ret != 0){
        if(iam == 0) printf("Failure on MLD2P4 precinit %d ictxt %d ptype %s\n",ret,LS_ICTXT_P(S),LS_PTYPE_P(S));
        return(SUNLS_PSET_FAIL_UNREC);
      }
  }
  return(SUNLS_SUCCESS);
}

int SUNLinSolSetup_PSBLAS(SUNLinearSolver S, SUNMatrix A){
  /* In this function we perform all the initialization for the solver and for
  the preconditioner. */
  psb_i_t ret;
  psb_i_t iam,np;

  if (S == NULL || A == NULL) return(SUNLS_MEM_NULL);
  psb_c_info(LS_ICTXT_P(S),&iam,&np);

  // Use the information contained in A to setup the field in S
  LS_DESCRIPTOR_P(S) = SM_DESCRIPTOR_P(A);
  LS_PMAT_P(S)       = SM_PMAT_P(A);
  if( LS_ICTXT_P(S) != SM_ICTXT_P(A)){
    printf("Working with different parallel context: this will never work.\n");
  }

  psb_c_info(LS_ICTXT_P(S),&iam,&np);
  if(iam==0) printf("I'm building the %s preconditioner\n",LS_PTYPE_P(S));

  // The init routine depends on the fact that we are using PSBLAS or MLD2P4
  if( strcmp(LS_PTYPE_P(S),"NONE") == 0 ||
      strcmp(LS_PTYPE_P(S),"BJAC") == 0 ||
      strcmp(LS_PTYPE_P(S),"DIAG") == 0 ){
      ret = psb_c_dprecbld(LS_PMAT_P(S),LS_DESCRIPTOR_P(S),LS_PREC_P(S));
      if(ret != 0) return(SUNLS_PSET_FAIL_UNREC);
  }else if(strcmp(LS_PTYPE_P(S),"ML") == 0 ||
    strcmp(LS_PTYPE_P(S),"GS") == 0 ||
    strcmp(LS_PTYPE_P(S),"AS") == 0 ||
    strcmp(LS_PTYPE_P(S),"FBGS") == 0 ){
      ret = mld_c_dhierarchy_build(LS_PMAT_P(S),LS_DESCRIPTOR_P(S),LS_MLPREC_P(S));
      if(ret != 0) return(SUNLS_PSET_FAIL_UNREC);
      ret = mld_c_dsmoothers_build(LS_PMAT_P(S),LS_DESCRIPTOR_P(S),LS_MLPREC_P(S));
      if(ret != 0) return(SUNLS_PSET_FAIL_UNREC);
  }

  return(SUNLS_SUCCESS);


}

int SUNLinSolSolve_PSBLAS(SUNLinearSolver S, SUNMatrix A,
                                            N_Vector x, N_Vector b,
                                            realtype tol){
  psb_i_t ret;

  PSBLAS_CONTENT(S)->options.eps = tol;
  N_VAsb_PSBLAS(b);
  N_VAsb_PSBLAS(x);
  /* Solve the linear system in PSBLAS, again we need to make a distinction
   * regarding the used preconditioner                                        */
  if( strcmp(LS_PTYPE_P(S),"NONE") == 0||
      strcmp(LS_PTYPE_P(S),"BJAC") == 0 ||
      strcmp(LS_PTYPE_P(S),"DIAG") == 0 ){
    ret=psb_c_dkrylov(LS_METHD_P(S),
                    LS_PMAT_P(S),
                    LS_PREC_P(S),
                    NV_PVEC_P(b),
                    NV_PVEC_P(x),
                    LS_DESCRIPTOR_P(S),
                    &(PSBLAS_CONTENT(S)->options));
    if(ret != 0) return(SUNLS_PACKAGE_FAIL_REC);
  }else if(strcmp(LS_PTYPE_P(S),"ML") == 0 ||
    strcmp(LS_PTYPE_P(S),"GS") == 0 ||
    strcmp(LS_PTYPE_P(S),"AS") == 0 ||
    strcmp(LS_PTYPE_P(S),"FBGS") == 0 ){
    ret=mld_c_dkrylov(LS_METHD_P(S),
                    LS_PMAT_P(S),
                    LS_MLPREC_P(S),
                    NV_PVEC_P(b),
                    NV_PVEC_P(x),
                    LS_DESCRIPTOR_P(S),
                    &(PSBLAS_CONTENT(S)->options));
    if(ret != 0) return(SUNLS_PACKAGE_FAIL_REC);
  }

  return(SUNLS_SUCCESS);
}

int SUNLinSolFree_PSBLAS(SUNLinearSolver S){
  /* This routine frees only the preconditioner and the structure containing the
  various part of the solver: the communicator and the matrix are still there,
  they should be freed after the matrix has been destroyed */

  if (S == NULL) return(SUNLS_SUCCESS);

  /* delete the preconditioner item from within the content structure */
  if( strcmp(LS_PTYPE_P(S),"NONE") == 0 ||
      strcmp(LS_PTYPE_P(S),"BJAC") == 0 ||
      strcmp(LS_PTYPE_P(S),"DIAG") == 0 ){
    psb_c_dprecfree(LS_PREC_P(S));
  }else if(strcmp(LS_PTYPE_P(S),"ML") == 0 ||
    strcmp(LS_PTYPE_P(S),"GS") == 0 ||
    strcmp(LS_PTYPE_P(S),"AS") == 0 ||
    strcmp(LS_PTYPE_P(S),"FBGS") == 0 ){
    mld_c_dprecfree(LS_MLPREC_P(S));
  }

  /* delete generic structures */
  free(S->content);  S->content = NULL;
  free(S->ops);  S->ops = NULL;
  free(S); S = NULL;
  return(SUNLS_SUCCESS);
}

int SUNLinSolNumIters_PSBLAS(SUNLinearSolver S){
  if (S == NULL) return(-ONE);
  return(PSBLAS_CONTENT(S)->options.iter);
}

realtype SUNLinSolResNorm_PSBLAS(SUNLinearSolver S){
  if (S == NULL) return(-ONE);
  return(PSBLAS_CONTENT(S)->options.err);
}

long int SUNLinSolLastFlag_PSBLAS(SUNLinearSolver S){
  if (S == NULL) return(-ONE);
  if (SUNLinSolResNorm_PSBLAS(S) <= PSBLAS_CONTENT(S)->options.eps ){
    return(SUNLS_SUCCESS);
  }else if (SUNLinSolResNorm_PSBLAS(S) > 1){
    return(SUNLS_CONV_FAIL);
  }else{
    return(SUNLS_RES_REDUCED);
  }
}

/* ---------------------------------------
 * Interfaces for PSBLAS/MLD2P4 routine for
 * setting up the preconditioner options in
 * SUNLinSol_PSBLAS
 * --------------------------------------- */

int SUNLinSolSeti_PSBLAS(SUNLinearSolver S, const char *what, psb_i_t val){
  if (S == NULL || LS_MLPREC_P(S) == NULL) return(-1);
  return(mld_c_dprecseti(LS_MLPREC_P(S), what, val));
}
int SUNLinSolSetc_PSBLAS(SUNLinearSolver S, const char *what, const char *val){
  if (S == NULL || LS_MLPREC_P(S) == NULL) return(-1);
  return(mld_c_dprecsetc(LS_MLPREC_P(S), what, val));
}
int SUNLinSolSetr_PSBLAS(SUNLinearSolver S, const char *what, double val){
  if (S == NULL || LS_MLPREC_P(S) == NULL) return(-1);
  return(mld_c_dprecsetr(LS_MLPREC_P(S), what, val));
}


#ifdef __cplusplus
}
#endif
