#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sunlinsol/sunlinsol_psblas.h>
#include <sundials/sundials_math.h>

#include "psb_base_cbind.h"
#include "psb_prec_cbind.h"
#include "psb_krylov_cbind.h"

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

#define PSBLAS_CONTENT(S)  ( (SUNLinearSolverContent_PSBLAS)(S->content) )

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif


/* Function used to create a PSBLAS linear solver */
SUNLinearSolver SUNLinSol_PSBLAS(psb_c_SolverOptions options, char methd[20], char ptype[20]){

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
  // TODO:Some of the function that are now set as null needs to be implemented!
  ops->gettype           = SUNLinSolGetType_PSBLAS;
  ops->setatimes         = NULL;
  ops->setpreconditioner = NULL;
  ops->setscalingvectors = NULL;
  ops->initialize        = SUNLinSolInitialize_PSBLAS;
  ops->setup             = SUNLinSolSetup_PSBLAS;
  ops->solve             = SUNLinSolSolve_PSBLAS;
  ops->numiters          = NULL;  // TODO
  ops->resnorm           = NULL;  // TODO
  ops->resid             = NULL;  // TODO
  ops->lastflag          = NULL;  // TODO
  ops->space             = NULL;  // TODO
  ops->free              = NULL;  // TODO

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_PSBLAS) malloc(sizeof(struct _SUNLinearSolverContent_PSBLAS));
  if (content == NULL) { free(ops); free(S); return(NULL); }

  /* Fill content */
  content->options=options;
  content->cdh=psb_c_new_descriptor();
  content->ictxt=psb_c_init();
  content->ph=psb_c_new_dprec();
  strcpy(content->methd,methd); // assignment to expression with array type!
  strcpy(content->ptype,ptype); // assignment to expression with array type!

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

  SUNLinearSolverContent_PSBLAS content;

  return(SUNLS_SUCCESS);
}

int SUNLinSolSetup_PSBLAS(SUNLinearSolver S, SUNMatrix A){
  /* In this function we perform all the initialization for the solver and for
  the preconditioner. We get the matrix SUNMatrix A and covert it into a PSBLAS
  matrix and we store it into the S structure. Moreover, we use it to setup the
  preconditioner ph in S. */

  if (S == NULL) return(SUNLS_MEM_NULL);

  psb_c_dspmat *ah;   // PSBLAS matrix will contain the copy of SUNMatrix A
  psb_i_t ret;

  // Convert the matrix :

  PSBLAS_CONTENT(S)->ah=ah;
  // Build the preconditioner
  psb_c_dprecinit(PSBLAS_CONTENT(S)->ictxt,
                  PSBLAS_CONTENT(S)->ph,
                  PSBLAS_CONTENT(S)->ptype);
  ret=psb_c_dprecbld(ah,PSBLAS_CONTENT(S)->cdh,PSBLAS_CONTENT(S)->ph);

  // check if the construction of the preconditioner has gone well and return
  if(ret == 0){
    return(SUNLS_PSET_FAIL_UNREC);
  }
  else{
    return(SUNLS_SUCCESS);
  }


}

int SUNLinSolSolve_PSBLAS(SUNLinearSolver S, SUNMatrix A,
                                            N_Vector x, N_Vector b,
                                            realtype tol){
  psb_i_t ret;
  psb_c_dvector *bh, *xh;

  PSBLAS_CONTENT(S)->options.eps = tol;
  // Convert the N_Vector b to PSBLAS format

  // Convert the N_Vector x to PSBLAS format

  // Solve the linear system in PSBLAS
  ret=psb_c_dkrylov(PSBLAS_CONTENT(S)->methd,
                    PSBLAS_CONTENT(S)->ah,
                    PSBLAS_CONTENT(S)->ph,
                    bh,xh,
                    PSBLAS_CONTENT(S)->cdh,&(PSBLAS_CONTENT(S)->options));


  if(ret == 0){
    // Copy back the solution to the N_Vector x
    // Put the various statistic in PSBLAS_CONTENT(S) for extraction
    return(SUNLS_SUCCESS);
  }
  else{
    return(SUNLS_PACKAGE_FAIL_REC); // Need more informative output here!
  }
}


#ifdef __cplusplus
}
#endif
