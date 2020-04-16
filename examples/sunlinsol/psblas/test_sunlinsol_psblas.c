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
 * This is the testing routine to check the SUNLinSol PSBLAS module
 * implementation.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>

#include <sundials/sundials_types.h>
#include <sundials/sundials_iterative.h>
#include <sunmatrix/sunmatrix_psblas.h>
#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include <sunlinsol/sunlinsol_psblas.h>

#include "test_sunlinsol.h"

#include "mpi.h"

int main(int argc, char *argv[])
{
  return(0);
}

int check_vector(N_Vector X, N_Vector Y, realtype tol)
{
  int failure = 0;
  realtype *xdata, *ydata;
  sunindextype xldata, yldata;
  sunindextype i;

  /* get vector data */
  xdata = N_VGetArrayPointer(X);
  ydata = N_VGetArrayPointer(Y);

  /* check data lengths */
  xldata = N_VGetLocalLength_PSBLAS(X);
  yldata = N_VGetLocalLength_PSBLAS(Y);

  if (xldata != yldata) {
    printf(">>> ERROR: check_vector: Different data array lengths \n");
    return(1);
  }

  /* check vector data */
  for(i=0; i < xldata; i++){
    failure += FNEQ(xdata[i], ydata[i], tol);
  }

  if (failure > ZERO)
    return(1);
  else
    return(0);
}
