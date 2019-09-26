#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>

#include "psb_base_cbind.h"
#include "psb_prec_cbind.h"
#include "psb_krylov_cbind.h"

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Structure containing the solver information */
struct _SUNLinearSolverContent_PSBLAS {
  psb_c_SolverOptions options; // Structure containing the PSBLAS solver options
  psb_c_dprec *ph; // PSBLAS preconditioner
  psb_c_descriptor *cdh; // PSBLAS descriptor
  psb_i_t ictxt; // PSBLAS context
  psb_c_dspmat *ah; // PSBLAS matrix
  char methd[40], ptype[20];
};

typedef struct _SUNLinearSolverContent_PSBLAS *SUNLinearSolverContent_PSBLAS;

/* Prototype for the function used to create a PSBLAS linear solver */
SUNDIALS_EXPORT SUNLinearSolver SUNLinSol_PSBLAS(psb_c_SolverOptions options, char methd[20], char ptype[20]);

/* Prototypes for the four required routines to define a linear solver */
SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolInitialize_PSBLAS(SUNLinearSolver S);
SUNDIALS_EXPORT int SUNLinSolSetup_PSBLAS(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_PSBLAS(SUNLinearSolver S, SUNMatrix A,
                                            N_Vector x, N_Vector b,
                                            realtype tol);

#ifdef __cplusplus
}
#endif
