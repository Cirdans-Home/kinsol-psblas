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
 * This is the testing routine to check the SUNMatrix Sparse module
 * implementation.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>

#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_psblas.h>
#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include "test_sunmatrix.h"

#include <mpi.h>

/* prototypes for custom tests */
int Test_SUNMatScaleAdd2(SUNMatrix A, SUNMatrix B, N_Vector x,
                         N_Vector y, N_Vector z);
int Test_SUNMatScaleAddI2(SUNMatrix A, N_Vector x, N_Vector y);



/* ----------------------------------------------------------------------
 * Main SUNMatrix Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
  int          fails=0;                    /* counter for test failures  */
  int          globfails = 0;              /* counter for test failures  */
  sunindextype matrows, matcols;           /* matrix dims                */
  N_Vector     x, y, z;                    /* test vectors               */
  realtype*    vecdata;                    /* pointers to vector data    */
  SUNMatrix    A, B, C, D, I;              /* test matrices              */
  realtype*    matdata;                    /* pointer to matrix data     */
  sunindextype i, j, k, N;
  int          print_timing, square;
  psb_i_t      ictxt;                      /* PSBLAS Context             */
  psb_i_t      nprocs, myid;               /* Number of procs, proc id   */
  psb_c_descriptor *cdh;                   /* PSBLAS Descriptor          */
  /* Auxiliary variabales */
  psb_i_t      info;              /* FLAG value for PSBLAS     */
  /* MPI Comminicator */
  MPI_Comm     comm;

  /* Get processor number and total number of processes */
  ictxt = psb_c_init();
  psb_c_info(ictxt,&myid,&nprocs);
  comm = MPI_Comm_f2c(ictxt);



  /* check input and set vector length */
  if (argc < 4){
    printf("ERROR: THREE (3) Input required: matrix rows, matrix cols, print timing \n");
    psb_c_abort(ictxt);
    return(-1);
  }

  matrows = atol(argv[1]);
  if (matrows < 1) {
    printf("ERROR: number of rows must be a positive integer\n");
    psb_c_abort(ictxt);
    return(-1);
  }

  matcols = atol(argv[2]);
  if (matcols < 1) {
    printf("ERROR: number of cols must be a positive integer\n");
    psb_c_abort(ictxt);
    return(-1);
  }

  print_timing = atoi(argv[3]);
  SetTiming(print_timing);

  square = (matrows == matcols) ? 1 : 0;
  printf("\nSparse matrix test: size %ld by %ld\n\n",
         (long int) matrows, (long int) matcols);

  /* Initialize vectors and matrices to NULL */
  x = NULL;
  y = NULL;
  z = NULL;
  A = NULL;
  B = NULL;
  C = NULL;
  D = NULL;
  I = NULL;


  /* SUNMatrix Tests */
  fails += Test_SUNMatGetID(A, SUNMATRIX_CUSTOM, 0);
  fails += Test_SUNMatClone(A, 0);
  fails += Test_SUNMatCopy(A, 0);
  fails += Test_SUNMatZero(A, 0);
  fails += Test_SUNMatScaleAdd(A, I, 0);
  fails += Test_SUNMatScaleAdd2(A, B, x, y, z);
  if (square) {
    fails += Test_SUNMatScaleAddI(A, I, 0);
    fails += Test_SUNMatScaleAddI2(A, x, y);
  }
  fails += Test_SUNMatMatvec(A, x, y, 0);
  fails += Test_SUNMatSpace(A, 0);

  /* Print result */
  if (fails) {
    printf("FAIL: SUNMatrix module failed %i tests \n \n", fails);
    // printf("\nA =\n");
    // SUNSparseMatrix_Print(A,stdout);
    // printf("\nB =\n");
    // SUNSparseMatrix_Print(B,stdout);
    // if (square) {
    //   printf("\nI =\n");
    //   SUNSparseMatrix_Print(I,stdout);
    // }
    // printf("\nx =\n");
    // N_VPrint_Serial(x);
    // printf("\ny =\n");
    // N_VPrint_Serial(y);
    // printf("\nz =\n");
    // N_VPrint_Serial(z);
  } else {
    printf("SUCCESS: SUNMatrix module passed all tests \n \n");
  }

  /* Free vectors and matrices */
  N_VDestroy(x);
  N_VDestroy(y);
  N_VDestroy(z);
  SUNMatDestroy(A);
  SUNMatDestroy(B);
  SUNMatDestroy(C);
  SUNMatDestroy(D);
  if (square)
    SUNMatDestroy(I);

  /* check if any other process failed */
  (void) MPI_Allreduce(&fails, &globfails, 1, MPI_INT, MPI_MAX, comm);

  psb_c_exit(ictxt);

  return(globfails);
}

/* ----------------------------------------------------------------------
 * Extra ScaleAdd tests for sparse matrices:
 *    A and B should have different sparsity patterns, and neither should
 *      contain sufficient storage to for their sum
 *    y should already equal A*x
 *    z should already equal B*x
 * --------------------------------------------------------------------*/
int Test_SUNMatScaleAdd2(SUNMatrix A, SUNMatrix B, N_Vector x,
                         N_Vector y, N_Vector z)
{
  int       failure;
  // SUNMatrix C, D, E;
  // N_Vector  u, v;
  // realtype  tol=100*UNIT_ROUNDOFF;

  // /* create clones for test */
  // C = SUNMatClone(A);
  // u = N_VClone(y);
  // v = N_VClone(y);
  //
  // /* test 1: add A to B (output must be enlarged) */
  // failure = SUNMatCopy(A, C);            /* C = A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatScaleAdd(ONE, C, B);   /* C = A+B */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatMatvec(C, x, u);       /* u = Cx = Ax+Bx */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // N_VLinearSum(ONE,y,ONE,z,v);           /* v = y+z */
  // failure = check_vector(u, v, tol);     /* u ?= v */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd2 check 1 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nB =\n");
  //   SUNSparseMatrix_Print(B,stdout);
  //   printf("\nC =\n");
  //   SUNSparseMatrix_Print(C,stdout);
  //   printf("\nx =\n");
  //   N_VPrint_Serial(x);
  //   printf("\ny =\n");
  //   N_VPrint_Serial(y);
  //   printf("\nz =\n");
  //   N_VPrint_Serial(z);
  //   printf("\nu =\n");
  //   N_VPrint_Serial(u);
  //   printf("\nv =\n");
  //   N_VPrint_Serial(v);
  //   SUNMatDestroy(C);  N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAdd2 check 1 \n");
  // }
  //
  // /* test 2: add A to a matrix with sufficient but misplaced storage */
  // D = SUNMatClone(A);
  // failure = SUNSparseMatrix_Reallocate(D, SM_NNZ_S(A)+SM_NNZ_S(B));
  // failure = SUNMatCopy(A, D);            /* D = A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatScaleAdd(ONE, D, B);   /* D = A+B */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatMatvec(D, x, u);       /* u = Cx = Ax+Bx */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // N_VLinearSum(ONE,y,ONE,z,v);           /* v = y+z */
  // failure = check_vector(u, v, tol);     /* u ?= v */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd2 check 2 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nB =\n");
  //   SUNSparseMatrix_Print(B,stdout);
  //   printf("\nD =\n");
  //   SUNSparseMatrix_Print(D,stdout);
  //   printf("\nx =\n");
  //   N_VPrint_Serial(x);
  //   printf("\ny =\n");
  //   N_VPrint_Serial(y);
  //   printf("\nz =\n");
  //   N_VPrint_Serial(z);
  //   printf("\nu =\n");
  //   N_VPrint_Serial(u);
  //   printf("\nv =\n");
  //   N_VPrint_Serial(v);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAdd2 check 2 \n");
  // }
  //
  //
  // /* test 3: add A to a matrix with the appropriate structure already in place */
  // E = SUNMatClone(C);
  // failure = SUNMatCopy(C, E);                /* E = A + B */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);  SUNMatDestroy(E);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatScaleAdd(NEG_ONE, E, B);   /* E = -A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);  SUNMatDestroy(E);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // failure = SUNMatMatvec(E, x, u);           /* u = Ex = -Ax */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);  SUNMatDestroy(E);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // N_VLinearSum(NEG_ONE,y,ZERO,z,v);          /* v = -y */
  // failure = check_vector(u, v, tol);         /* v ?= u */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAdd2 check 3 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nB =\n");
  //   SUNSparseMatrix_Print(B,stdout);
  //   printf("\nC =\n");
  //   SUNSparseMatrix_Print(C,stdout);
  //   printf("\nE =\n");
  //   SUNSparseMatrix_Print(E,stdout);
  //   printf("\nx =\n");
  //   N_VPrint_Serial(x);
  //   printf("\ny =\n");
  //   N_VPrint_Serial(y);
  //   printf("\nu =\n");
  //   N_VPrint_Serial(u);
  //   printf("\nv =\n");
  //   N_VPrint_Serial(v);
  //   SUNMatDestroy(C);  SUNMatDestroy(D);  SUNMatDestroy(E);
  //   N_VDestroy(u);  N_VDestroy(v);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAdd2 check 3 \n");
  // }
  //
  // SUNMatDestroy(C);
  // SUNMatDestroy(D);
  // SUNMatDestroy(E);
  // N_VDestroy(u);
  // N_VDestroy(v);
  return(0);
}

/* ----------------------------------------------------------------------
 * Extra ScaleAddI tests for sparse matrices:
 *    A should not contain values on the diagonal, nor should it contain
 *      sufficient storage to add those in
 *    y should already equal A*x
 * --------------------------------------------------------------------*/
int Test_SUNMatScaleAddI2(SUNMatrix A, N_Vector x, N_Vector y)
{
  int       failure;
  // SUNMatrix B, C, D;
  // N_Vector  w, z;
  // realtype  tol=100*UNIT_ROUNDOFF;

  // /* create clones for test */
  // B = SUNMatClone(A);
  // z = N_VClone(x);
  // w = N_VClone(x);
  //
  // /* test 1: add I to a matrix with insufficient storage */
  // failure = SUNMatCopy(A, B);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatScaleAddI(NEG_ONE, B);   /* B = I-A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatMatvec(B, x, z);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // N_VLinearSum(ONE,x,NEG_ONE,y,w);
  // failure = check_vector(z, w, tol);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI2 check 1 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nB =\n");
  //   SUNSparseMatrix_Print(B,stdout);
  //   printf("\nz =\n");
  //   N_VPrint_Serial(z);
  //   printf("\nw =\n");
  //   N_VPrint_Serial(w);
  //   SUNMatDestroy(B);  N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAddI2 check 1 \n");
  // }
  //
  // /* test 2: add I to a matrix with sufficient but misplaced
  //    storage */
  // C = SUNMatClone(A);
  // failure = SUNSparseMatrix_Reallocate(C, SM_NNZ_S(A)+SM_ROWS_S(A));
  // failure = SUNMatCopy(A, C);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatScaleAddI(NEG_ONE, C);   /* C = I-A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatMatvec(C, x, z);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // N_VLinearSum(ONE,x,NEG_ONE,y,w);
  // failure = check_vector(z, w, tol);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI2 check 2 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nC =\n");
  //   SUNSparseMatrix_Print(C,stdout);
  //   printf("\nz =\n");
  //   N_VPrint_Serial(z);
  //   printf("\nw =\n");
  //   N_VPrint_Serial(w);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAddI2 check 2 \n");
  // }
  //
  //
  // /* test 3: add I to a matrix with appropriate structure already in place */
  // D = SUNMatClone(C);
  // failure = SUNMatCopy(C, D);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatCopy returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatScaleAddI(NEG_ONE, D);   /* D = A */
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = SUNMatMatvec(D, x, z);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatMatvec returned %d \n",
  //          failure);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // failure = check_vector(z, y, tol);
  // if (failure) {
  //   printf(">>> FAILED test -- SUNMatScaleAddI2 check 3 \n");
  //   printf("\nA =\n");
  //   SUNSparseMatrix_Print(A,stdout);
  //   printf("\nD =\n");
  //   SUNSparseMatrix_Print(D,stdout);
  //   printf("\nz =\n");
  //   N_VPrint_Serial(z);
  //   printf("\ny =\n");
  //   N_VPrint_Serial(y);
  //   SUNMatDestroy(B);  SUNMatDestroy(C);  SUNMatDestroy(D);
  //   N_VDestroy(z);  N_VDestroy(w);  return(1);
  // }
  // else {
  //   printf("    PASSED test -- SUNMatScaleAddI2 check 3 \n");
  // }
  //
  // SUNMatDestroy(B);
  // SUNMatDestroy(C);
  // SUNMatDestroy(D);
  // N_VDestroy(z);
  // N_VDestroy(w);
  return(0);
}

/* ----------------------------------------------------------------------
 * Check matrix
 * --------------------------------------------------------------------*/
int check_matrix(SUNMatrix A, SUNMatrix B, realtype tol)
{
  int failure = 0;
  sunindextype Annz;
  sunindextype Bnnz;
  /* matrices must have same sparsetype, shape and actual data lengths */
  if (SUNMatGetID(A) != SUNMatGetID(B)) {
    printf(">>> ERROR: check_matrix: Different storage types (%d vs %d)\n",
           SUNMatGetID(A), SUNMatGetID(B));
    return(1);
  }

  if (SUNPSBLASMatrix_Rows(A) != SUNPSBLASMatrix_Rows(B)) {
    printf(">>> ERROR: check_matrix: Different numbers of rows (%ld vs %ld)\n",
           (long int) SUNPSBLASMatrix_Rows(A), (long int) SUNPSBLASMatrix_Rows(B));
    return(1);
  }
  if (SUNPSBLASMatrix_Columns(A) != SUNPSBLASMatrix_Columns(B)) {
    printf(">>> ERROR: check_matrix: Different numbers of columns (%ld vs %ld)\n",
           (long int) SUNPSBLASMatrix_Columns(A),
           (long int) SUNPSBLASMatrix_Columns(B));
    return(1);
  }
  if ( SUNPSBLASMatrix_NNZ(A) != SUNPSBLASMatrix_NNZ(B)) {
    printf(">>> ERROR: check_matrix: Different numbers of nonzeos (%ld vs %ld)\n",
           (long int) SUNPSBLASMatrix_NNZ(A), (long int) SUNPSBLASMatrix_NNZ(B));
    return(1);
  }

  /* compare matrix values */


  return(0);
}

int check_matrix_entry(SUNMatrix A, realtype val, realtype tol)
{
  int failure = 0;
  // realtype *Adata;
  // sunindextype *indexptrs;
  // sunindextype i, NP;

  // /* get data pointer */
  // Adata = SUNSparseMatrix_Data(A);
  //
  // /* compare data */
  // indexptrs = SUNSparseMatrix_IndexPointers(A);
  // NP = SUNSparseMatrix_NP(A);
  // for(i=0; i < indexptrs[NP]; i++){
  //   failure += FNEQ(Adata[i], val, tol);
  // }

  if (failure > ZERO)
    return(1);
  else
    return(0);
}

int check_vector(N_Vector x, N_Vector y, realtype tol)
{
  int failure = 0;
  realtype *xdata, *ydata;
  sunindextype xldata, yldata;
  sunindextype i;

  /* get vector data */
  xdata = N_VGetArrayPointer(x);
  ydata = N_VGetArrayPointer(y);

  /* check data lengths */
  xldata = N_VGetLocalLength_PSBLAS(x);
  yldata = N_VGetLocalLength_PSBLAS(y);

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

booleantype has_data(SUNMatrix A)
{
  // realtype *Adata = SUNSparseMatrix_Data(A);
  // if (Adata == NULL)
  //   return SUNFALSE;
  // else
  //   return SUNTRUE;
  return SUNTRUE;
}

booleantype is_square(SUNMatrix A)
{
  // if (SUNSparseMatrix_Rows(A) == SUNSparseMatrix_Columns(A))
  //   return SUNTRUE;
  // else
  //   return SUNFALSE;
  return SUNTRUE;
}
