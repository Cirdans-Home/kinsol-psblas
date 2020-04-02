/* -----------------------------------------------------------------
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
 * This is the testing routine to check the NVECTOR PSBLAS module
 * implementation.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include <sundials/sundials_types.h>
#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include "test_nvector.h"

#include <mpi.h>

/* ----------------------------------------------------------------------
 * Main NVector Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
  int          fails = 0;         /* counter for test failures */
  int          globfails = 0;     /* counter for test failures */
  int          retval;            /* function return value     */
  sunindextype local_length;      /* local vector length       */
  sunindextype global_length;     /* global vector length      */
  N_Vector     U, V, W, X, Y, Z;  /* test vectors              */
  int          print_timing;      /* turn timing on/off        */
  psb_i_t      ictxt;             /* PSBLAS Context            */
  psb_i_t      nprocs, myid;      /* Number of procs, proc id  */
  psb_c_descriptor *cdh;          /* PSBLAS Descriptor         */
  /* Auxiliary variabales */
  psb_i_t      info;              /* FLAG value for PSBLAS     */
  psb_l_t      ng;                /* Global length             */
  psb_i_t      nb, nl;            /* Local lenth               */
  psb_l_t      *vl;               /* Vector needed for descritor construction */
  psb_l_t      i;
  /* MPI Comminicator */
  MPI_Comm     comm;

  /* Get processor number and total number of processes */
  ictxt = psb_c_init();
  psb_c_info(ictxt,&myid,&nprocs);
  comm = MPI_Comm_f2c(ictxt);

  /* check inputs */
  if (argc < 3) {
    if (myid == 0)
      printf("ERROR: TWO (2) Inputs required: vector length, print timing \n");
    psb_c_abort(ictxt);
  }

  local_length = atol(argv[1]);
  if (local_length < 1) {
    if (myid == 0)
      printf("ERROR: local vector length must be a positive integer \n");
    psb_c_abort(ictxt);
  }

  print_timing = atoi(argv[2]);
  SetTiming(print_timing, myid);

  /* global length */
  global_length = nprocs*local_length;

  /* To work with the PSBLAS library, we need to build a descriptor that
  encodes the DATA DISTRIBUTION, we assume here a simple minded BLOCK
  data distribution */
  cdh=psb_c_new_descriptor();
  psb_c_set_index_base(0);

  // ng = (psb_l_t) global_length;
  // nb = nprocs;
  // nl = nb;
  // if ( (ng -myid*nb) < nl) nl = ng -myid*nb;
  // if ((vl=malloc(nb*sizeof(psb_l_t)))==NULL) {
  //   fprintf(stderr,"On %d: malloc failure\n",myid);
  //   psb_c_abort(ictxt);
  // }
  // i = ((psb_l_t)myid) * nb;
  // for (int k=0; k<nl; k++)
  //   vl[k] = i+k;
  //
  // if ((info=psb_c_cdall_vl(nl,vl,ictxt,cdh))!=0) {
  //   fprintf(stderr,"From cdall: %d\nBailing out\n",info);
  //   psb_c_abort(ictxt);
  // }

  if (info=psb_c_cdall_nl(local_length, ictxt, cdh)!=0) {
    fprintf(stderr,"From cdall: %d\nBailing out\n",info);
    psb_c_abort(ictxt);
  }

  if ((info=psb_c_cdasb(cdh))!=0)  return(info);

  if (myid == 0) {
    printf("-- PSBLAS descriptor allocated -- \n");
    printf("Number of global rows : %ld\n", psb_c_cd_get_global_rows(cdh));
    printf("Number of local rows : %d\n\n", psb_c_cd_get_local_rows(cdh));
    printf("Testing the parallel PSBLAS N_Vector \n");
    printf("Vector global length %ld \n", (long int) global_length);
    printf("MPI processes %d \n", nprocs);
  }

  /* Create new vectors */
  W = N_VNewEmpty_PSBLAS(ictxt, cdh);
  if (W == NULL) {
    if (myid == 0) printf("FAIL: Unable to create a new empty vector \n\n");
    psb_c_abort(ictxt);
  }
  //
  //
  X = N_VNew_PSBLAS(ictxt, cdh);
  if (X == NULL) {
    N_VDestroy_PSBLAS(W);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
      psb_c_abort(ictxt);
  }

  /* Check vector ID */
  fails += Test_N_VGetVectorID(X, SUNDIALS_NVEC_CUSTOM, myid);

  /* Test clone functions */
  fails += Test_N_VCloneEmpty(X, myid);
  fails += Test_N_VClone(X, local_length, myid);
  fails += Test_N_VCloneEmptyVectorArray(5, X, myid);
  fails += Test_N_VCloneVectorArray(5, X, local_length, myid);

  /* Test setting/getting array data */
  // fails += Test_N_VSetArrayPointer(W, local_length, myid);
  fails += Test_N_VGetArrayPointer(X, local_length, myid);

  /* Clone additional vectors for testing */
  Y = N_VClone_PSBLAS(X);
  if (Y == NULL) {
    N_VDestroy_PSBLAS(W);
    N_VDestroy_PSBLAS(X);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
    psb_c_abort(ictxt);
  }

  Z = N_VClone_PSBLAS(X);
  if (Z == NULL) {
    N_VDestroy_PSBLAS(W);
    N_VDestroy_PSBLAS(X);
    N_VDestroy_PSBLAS(Y);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
    psb_c_abort(ictxt);
  }

  /* Standard vector operation tests */
  if (myid == 0) printf("\nTesting standard vector operations:\n\n");

  fails += Test_N_VConst(X, local_length, myid);
  fails += Test_N_VLinearSum(X, Y, Z, local_length, myid);
  fails += Test_N_VProd(X, Y, Z, local_length, myid);
  fails += Test_N_VDiv(X, Y, Z, local_length, myid);
  fails += Test_N_VScale(X, Z, local_length, myid);
  fails += Test_N_VAbs(X, Z, local_length, myid);
  fails += Test_N_VInv(X, Z, local_length, myid);
  fails += Test_N_VAddConst(X, Z, local_length, myid);
  fails += Test_N_VDotProd(X, Y, local_length, global_length, myid);
  fails += Test_N_VMaxNorm(X, local_length, myid);
  fails += Test_N_VWrmsNorm(X, Y, local_length, myid);
  fails += Test_N_VWrmsNormMask(X, Y, Z, local_length, global_length, myid);
  fails += Test_N_VMin(X, local_length, myid);
  fails += Test_N_VWL2Norm(X, Y, local_length, global_length, myid);
  fails += Test_N_VL1Norm(X, local_length, global_length, myid);
  fails += Test_N_VCompare(X, Z, local_length, myid);
  fails += Test_N_VInvTest(X, Z, local_length, myid);
  fails += Test_N_VConstrMask(X, Y, Z, local_length, myid);
  fails += Test_N_VMinQuotient(X, Y, local_length, myid);

  /* Fused and vector array operations tests (disabled) */
  if (myid == 0) printf("\nTesting fused and vector array operations (disabled):\n\n");

  /* create vector and disable all fused and vector array operations */
  U = N_VNew_PSBLAS(ictxt, cdh);
  retval = N_VEnableFusedOps_PSBLAS(U, SUNFALSE);
  if (U == NULL || retval != 0) {
    N_VDestroy_PSBLAS(W);
    N_VDestroy_PSBLAS(X);
    N_VDestroy_PSBLAS(Y);
    N_VDestroy_PSBLAS(Z);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
    psb_c_abort(ictxt);
  }

  /* fused operations */
  fails += Test_N_VLinearCombination(U, local_length, myid);
  fails += Test_N_VScaleAddMulti(U, local_length, myid);
  fails += Test_N_VDotProdMulti(U, local_length, global_length, myid);

  /* vector array operations */
  fails += Test_N_VLinearSumVectorArray(U, local_length, myid);
  fails += Test_N_VScaleVectorArray(U, local_length, myid);
  fails += Test_N_VConstVectorArray(U, local_length, myid);
  fails += Test_N_VWrmsNormVectorArray(U, local_length, myid);
  fails += Test_N_VWrmsNormMaskVectorArray(U, local_length, global_length, myid);
  fails += Test_N_VScaleAddMultiVectorArray(U, local_length, myid);
  fails += Test_N_VLinearCombinationVectorArray(U, local_length, myid);

  /* Fused and vector array operations tests (enabled) */
  if (myid == 0) printf("\nTesting fused and vector array operations (enabled):\n\n");

  /* create vector and enable all fused and vector array operations */
  V = N_VNew_PSBLAS(ictxt, cdh);
  retval = N_VEnableFusedOps_PSBLAS(V, SUNTRUE);
  if (V == NULL || retval != 0) {
    N_VDestroy_PSBLAS(W);
    N_VDestroy_PSBLAS(X);
    N_VDestroy_PSBLAS(Y);
    N_VDestroy_PSBLAS(Z);
    N_VDestroy_PSBLAS(U);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
    psb_c_abort(ictxt);
  }

  /* fused operations */
  // fails += Test_N_VLinearCombination(V, local_length, myid);
  fails += Test_N_VScaleAddMulti(V, local_length, myid);
  fails += Test_N_VDotProdMulti(V, local_length, global_length, myid);

  /* vector array operations */
  fails += Test_N_VLinearSumVectorArray(V, local_length, myid);
  fails += Test_N_VScaleVectorArray(V, local_length, myid);
  fails += Test_N_VConstVectorArray(V, local_length, myid);
  fails += Test_N_VWrmsNormVectorArray(V, local_length, myid);
  fails += Test_N_VWrmsNormMaskVectorArray(V, local_length, global_length, myid);
  fails += Test_N_VScaleAddMultiVectorArray(V, local_length, myid);
  fails += Test_N_VLinearCombinationVectorArray(V, local_length, myid);

  /* Free vectors */
  N_VDestroy_PSBLAS(W);
  N_VDestroy_PSBLAS(X);
  N_VDestroy_PSBLAS(Y);
  N_VDestroy_PSBLAS(Z);
  N_VDestroy_PSBLAS(U);
  N_VDestroy_PSBLAS(V);

  /* Print result */
  if (fails) {
    printf("FAIL: NVector module failed %i tests, Proc %d \n\n", fails, myid);
  } else {
    if (myid == 0)
      printf("SUCCESS: NVector_PSBLAS module passed all tests \n\n");
  }

  /* check if any other process failed */
  (void) MPI_Allreduce(&fails, &globfails, 1, MPI_INT, MPI_MAX, comm);

  psb_c_exit(ictxt);

  return(globfails);
}

/* ----------------------------------------------------------------------
 * Implementation specific utility functions for vector tests
 * --------------------------------------------------------------------*/
int check_ans(realtype ans, N_Vector X, sunindextype local_length)
{
  int          failure = 0;
  sunindextype i;
  realtype     *Xdata;

  Xdata = N_VGetArrayPointer(X);

  /* check vector data */
  for (i = 0; i < local_length; i++) {
    failure += FNEQ(Xdata[i], ans);
  }

  return (failure > ZERO) ? (1) : (0);
}

booleantype has_data(N_Vector X)
{
  /* check if data array is non-null */
  return (N_VGetArrayPointer(X) == NULL) ? SUNFALSE : SUNTRUE;
}

void set_element(N_Vector X, sunindextype i, realtype val)
{
  psb_l_t irow[1];
  double value[1];
  psb_i_t myid,nprocs;

  psb_c_info(NV_ICTXT_P(X),&myid,&nprocs);
  /* set i-th element of data array */
  irow[0] = myid*N_VGetLocalLength_PSBLAS(X) + i;
  value[0] = val;
  psb_c_dgeins(1,irow,value,NV_PVEC_P(X),NV_DESCRIPTOR_P(X));
  N_VAsb_PSBLAS(X);
}

realtype get_element(N_Vector X, sunindextype i)
{
  /* get i-th element of data array */
  return (N_VGetArrayPointer_PSBLAS(X))[i];
}

double max_time(N_Vector X, double time)
{
  double maxt;
  MPI_Comm comm;

  comm = MPI_Comm_f2c(NV_ICTXT_P(X));

  /* get max time across all MPI ranks */
  (void) MPI_Reduce(&time, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  return(maxt);
}

void sync_device()
{
  /* not running on GPU, just return */
  return;
}
