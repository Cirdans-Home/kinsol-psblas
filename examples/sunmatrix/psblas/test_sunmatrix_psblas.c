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

#define NBMAX       20

/* prototypes for PSBLAS matrix generation */
double  a1(double x, double y, double  z);
double  a2(double x, double y, double  z);
double  a3(double x, double y, double  z);
double  b1(double x, double y, double  z);
double  b2(double x, double y, double  z);
double  b3(double x, double y, double  z);
double 	g(double x, double y, double z);
psb_i_t matgen(psb_i_t ictxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A);
psb_i_t eyegen(psb_i_t ictxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A);

/* ----------------------------------------------------------------------
 * Main SUNMatrix PSBLAS Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
  int          fails=0;                    /* counter for test failures  */
  int          globfails = 0;              /* counter for test failures  */
  sunindextype matrows, matcols;           /* matrix dims                */
  N_Vector     x, y ;                    	 /* test vectors               */
  SUNMatrix    A, B, I;            				 /* test matrices              */
  int          print_timing, square;
  psb_i_t      ictxt;                      /* PSBLAS Context             */
  psb_i_t      nprocs, myid;               /* Number of procs, proc id   */
  psb_c_descriptor *cdh;                   /* PSBLAS Descriptor          */
  /* Auxiliary variabales */
  psb_i_t      info;              				/* FLAG value for PSBLAS     */
  MPI_Comm     comm;											/* MPI Comminicator */
	psb_i_t nb,nlr,nl,idim;									/* Poisson problem variables */
	psb_l_t i,ng, *vl, k;

	realtype  tol=10*UNIT_ROUNDOFF;

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
	if(myid == 0){
		printf("\nSparse matrix test: size %ld^3 by %ld^3\n\n",
         	(long int) matrows, (long int) matcols);
	}

	cdh = psb_c_new_descriptor();
	psb_c_set_index_base(0);
	idim = matrows;
	/* Simple minded BLOCK data distribution */
	ng = ((psb_l_t) idim)*idim*idim;
	nb = (ng+nprocs-1)/nprocs;
	nl = nb;
	if ( (ng -myid*nb) < nl) nl = ng -myid*nb;
	fprintf(stderr,"%d: Input data %d %ld %d %d\n",myid,idim,ng,nb, nl);
	vl = malloc(nb*sizeof(psb_l_t));
	if (vl == NULL) {
	  fprintf(stderr,"On %d: malloc failure\n",myid);
	  psb_c_abort(ictxt);
	}
	i = ((psb_l_t)myid) * nb;
	for (k=0; k<nl; k++)
	  vl[k] = i+k;

	info=psb_c_cdall_vl(nl,vl,ictxt,cdh);
	if (info != 0) {
	  fprintf(stderr,"From cdall: %d\nBailing out\n",info);
	  psb_c_abort(ictxt);
	}
	if(myid == 0) printf("Descriptor for square problem allocated.\n");
  /* Initialize vectors and matrices to NULL */
  x = NULL;
  y = NULL;
  A = NULL;
	B = NULL;
  I = NULL;
	/* Allocate the space for everything */
	A = SUNPSBLASMatrix(ictxt, cdh);
	if (A == NULL) {
		if (myid == 0) printf("FAIL: Unable to create a new matrix \n\n");
			psb_c_abort(ictxt);
			return(1);
	}
	B = SUNPSBLASMatrix(ictxt, cdh);
	if (B == NULL) {
		SUNMatDestroy_PSBLAS(A);
		if (myid == 0) printf("FAIL: Unable to create a new matrix \n\n");
			psb_c_abort(ictxt);
			return(1);
	}
	I = SUNPSBLASMatrix(ictxt, cdh);
	if (I == NULL){
		SUNMatDestroy_PSBLAS(A);
		SUNMatDestroy_PSBLAS(B);
		if (myid == 0) printf("FAIL: Unable to create a new matrix \n\n");
			psb_c_abort(ictxt);
			return(1);
	}
	x = N_VNew_PSBLAS(ictxt, cdh);
	if (x == NULL) {
		SUNMatDestroy_PSBLAS(A);
		SUNMatDestroy_PSBLAS(B);
		SUNMatDestroy_PSBLAS(I);
		if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
			psb_c_abort(ictxt);
			return(1);
	}
	y = N_VNew_PSBLAS(ictxt, cdh);
	if (y == NULL) {
		SUNMatDestroy_PSBLAS(A);
		SUNMatDestroy_PSBLAS(B);
		SUNMatDestroy_PSBLAS(I);
		N_VDestroy_PSBLAS(x);
		if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
			psb_c_abort(ictxt);
			return(1);
	}
	if (matgen(ictxt, nl, idim, vl,A)!= 0) {
    fprintf(stderr,"Error during matrix build loop for A\n");
    psb_c_abort(ictxt);
		return(1);
  }
	if(myid == 0) printf("Created the PSBLAS A matrix.\n");
	if (eyegen(ictxt, nl, idim, vl,I)!= 0) {
		fprintf(stderr,"Error during matrix build loop for I\n");
		psb_c_abort(ictxt);
		return(1);
	}
	if(myid == 0) printf("Created the PSBLAS I matrix.\n");

	info=psb_c_cdasb(cdh);
	if (info!=0)  return(info);

	N_VConst_PSBLAS(1.0,x);
	N_VConst_PSBLAS(1.0,y);
	SUNMatAsb_PSBLAS(A);
	SUNMatAsb_PSBLAS(B);
	SUNMatAsb_PSBLAS(I);

	SUNMatCopy(I,B); // B = eye matrix

	psb_c_barrier(ictxt);

	if(myid == 0) printf("All the PSBLAS objects have been populated.\n");
  /* SUNMatrix Tests */
  fails += Test_SUNMatGetID(A, SUNMATRIX_CUSTOM, myid);
	fails += Test_SUNMatClone(A, myid);
  fails += Test_SUNMatCopy(A, myid);
  fails += Test_SUNMatZero(A, myid);
  fails += Test_SUNMatScaleAdd(A, I, myid);
  if (square) {
   fails += Test_SUNMatScaleAddI(A, I, myid);
  }
  fails += Test_SUNMatMatvec(B, x, y, myid);
  fails += Test_SUNMatSpace(A, myid);

	/* Free vectors and matrices */
  N_VDestroy_PSBLAS(x);
  N_VDestroy_PSBLAS(y);
  SUNMatDestroy_PSBLAS(A);
	SUNMatDestroy_PSBLAS(B);
  SUNMatDestroy_PSBLAS(I);

	psb_c_barrier(ictxt);
	/* Print result */
  if (fails) {
    printf("FAIL: SUNMATRIX module failed %i tests, Proc %d \n\n", fails, myid);
  } else {
    if (myid == 0)
      printf("SUCCESS: SUNMATRIX_PSBLAS module passed all tests \n\n");
  }

  /* check if any other process failed */
  (void) MPI_Allreduce(&fails, &globfails, 1, MPI_INT, MPI_MAX, comm);


	if ((info=psb_c_cdfree(cdh))!=0) {
    fprintf(stderr,"From cdfree: %d\nBailing out\n",info);
    psb_c_abort(ictxt);
  }

	free(cdh);

	psb_c_barrier(ictxt);
	if (myid == 0) fprintf(stderr,"Test program completed successfully\n");
  psb_c_exit(ictxt);

  return(globfails);
}

/* ----------------------------------------------------------------------
 * Check matrix
 * --------------------------------------------------------------------*/
psb_i_t check_matrix(SUNMatrix A, SUNMatrix B, realtype tol)
{
	bool test;
  sunindextype Annz;
  sunindextype Bnnz;

	/* matrices should be in the ASSEMBLED state */
	if(!psb_c_dis_matasb(SM_PMAT_P(A),SM_DESCRIPTOR_P(A)))
		psb_c_dspasb(SM_PMAT_P(A),SM_DESCRIPTOR_P(A));
	if(!psb_c_dis_matasb(SM_PMAT_P(B),SM_DESCRIPTOR_P(B)))
		psb_c_dspasb(SM_PMAT_P(B),SM_DESCRIPTOR_P(B));

  /* matrices must have same shape and actual data */
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
  test = psb_c_dgecmpmat(SM_PMAT_P(A),SM_PMAT_P(B),tol,SM_DESCRIPTOR_P(A));

	if(test){
		return(0);
	}else{
		return(1);
	}

}

int check_matrix_entry(SUNMatrix A, realtype val, realtype tol)
{
  bool test;

	/* matrix should be in the ASSEMBLED state */
	if(!psb_c_dis_matasb(SM_PMAT_P(A),SM_DESCRIPTOR_P(A)))
		psb_c_dspasb(SM_PMAT_P(A),SM_DESCRIPTOR_P(A));

  test = psb_c_dgecmpmat_val(SM_PMAT_P(A),val,tol,SM_DESCRIPTOR_P(A));

  if (test)
    return(0);
  else
    return(1);
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
  if (SM_PMAT_P(A) == NULL)
    return SUNFALSE;
  else
    return SUNTRUE;
}

booleantype is_square(SUNMatrix A)
{
  if (SUNPSBLASMatrix_Rows(A) == SUNPSBLASMatrix_Columns(A))
     return SUNTRUE;
  else
     return SUNFALSE;
}


double  a1(double x, double y, double  z)
{
  return(1.0/80.0);
}
double a2(double x, double y, double  z)
{
  return(1.0/80.0);
}
double a3(double x, double y, double  z)
{
  return(1.0/80.0);
}
double  c(double x, double y, double  z)
{
  return(0.0);
}
double  b1(double x, double y, double  z)
{
  return(0.0/sqrt(3.0));
}
double b2(double x, double y, double  z)
{
  return(0.0/sqrt(3.0));
}
double b3(double x, double y, double  z)
{
  return(0.0/sqrt(3.0));
}

double g(double x, double y, double z)
{
  if (x == 1.0) {
    return(1.0);
  } else if (x == 0.0) {
    return( exp(-y*y-z*z));
  } else {
    return(0.0);
  }
}

psb_i_t matgen(psb_i_t ictxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A)
{
  psb_i_t iam, np;
  psb_l_t ix, iy, iz, el,glob_row;
  psb_i_t i, k, info;
  double x, y, z, deltah, sqdeltah, deltah2;
  double val[10*NBMAX], zt[NBMAX];
  psb_l_t irow[10*NBMAX], icol[10*NBMAX];

  info = 0;
  psb_c_info(ictxt,&iam,&np);
  deltah = (double) 1.0/(idim+1);
  sqdeltah = deltah*deltah;
  deltah2  = 2.0* deltah;
  psb_c_set_index_base(0);
  for (i=0; i<nl;  i++) {
    glob_row=vl[i];
    el=0;
    ix = glob_row/(idim*idim);
    iy = (glob_row-ix*idim*idim)/idim;
    iz = glob_row-ix*idim*idim-iy*idim;
    x=(ix+1)*deltah;
    y=(iy+1)*deltah;
    z=(iz+1)*deltah;
    zt[0] = 0.0;
    /*  internal point: build discretization */
    /*  term depending on   (x-1,y,z)        */
    val[el] = -a1(x,y,z)/sqdeltah-b1(x,y,z)/deltah2;
    if (ix==0) {
      zt[0] += g(0.0,y,z)*(-val[el]);
    } else {
      icol[el]=(ix-1)*idim*idim+(iy)*idim+(iz);
      el=el+1;
    }
    /*  term depending on     (x,y-1,z) */
    val[el]  = -a2(x,y,z)/sqdeltah-b2(x,y,z)/deltah2;
    if (iy==0) {
      zt[0] += g(x,0.0,z)*(-val[el]);
    } else {
      icol[el]=(ix)*idim*idim+(iy-1)*idim+(iz);
      el=el+1;
    }
    /* term depending on     (x,y,z-1)*/
    val[el]=-a3(x,y,z)/sqdeltah-b3(x,y,z)/deltah2;
    if (iz==0) {
      zt[0] += g(x,y,0.0)*(-val[el]);
    } else {
      icol[el]=(ix)*idim*idim+(iy)*idim+(iz-1);
      el=el+1;
    }
    /* term depending on     (x,y,z)*/
    val[el]=2.0*(a1(x,y,z)+a2(x,y,z)+a3(x,y,z))/sqdeltah + c(x,y,z);
    icol[el]=(ix)*idim*idim+(iy)*idim+(iz);
    el=el+1;
    /*  term depending on     (x,y,z+1) */
    val[el] = -a3(x,y,z)/sqdeltah+b3(x,y,z)/deltah2;
    if (iz==idim-1) {
      zt[0] += g(x,y,1.0)*(-val[el]);
    } else {
      icol[el]=(ix)*idim*idim+(iy)*idim+(iz+1);
      el=el+1;
    }
    /* term depending on     (x,y+1,z) */
    val[el] = -a2(x,y,z)/sqdeltah+b2(x,y,z)/deltah2;
    if (iy==idim-1) {
      zt[0] += g(x,1.0,z)*(-val[el]);
    } else {
      icol[el]=(ix)*idim*idim+(iy+1)*idim+(iz);
      el=el+1;
    }
    /*  term depending on     (x+1,y,z) */
    val[el] = -a1(x,y,z)/sqdeltah+b1(x,y,z)/deltah2;
    if (ix==idim-1) {
      zt[0] += g(1.0,y,z)*(-val[el]);
    } else {
      icol[el]=(ix+1)*idim*idim+(iy)*idim+(iz);
      el=el+1;
    }
    for (k=0; k<el; k++) irow[k]=glob_row;
    if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(A),SM_DESCRIPTOR_P(A)))!=0)
      fprintf(stderr,"From psb_c_dspins: %d\n",info);
  }

  return(info);


}

psb_i_t eyegen(psb_i_t ictxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A)
{
  psb_i_t iam, np;
  psb_l_t ix, iy, iz, el,glob_row;
  psb_i_t i, k, info;
  double x, y, z, deltah;
  double val[10*NBMAX], zt[NBMAX];
  psb_l_t irow[10*NBMAX], icol[10*NBMAX];

  info = 0;
  psb_c_info(ictxt,&iam,&np);
  deltah = (double) 1.0/(idim+1);
  psb_c_set_index_base(0);
  for (i=0; i<nl;  i++) {
    glob_row=vl[i];
    el=0;
    ix = glob_row/(idim*idim);
    iy = (glob_row-ix*idim*idim)/idim;
    iz = glob_row-ix*idim*idim-iy*idim;
    x=(ix+1)*deltah;
    y=(iy+1)*deltah;
    z=(iz+1)*deltah;
    zt[0] = 0.0;
    /*  internal point: build discretization */
    val[el]=1.0;
    icol[el]=(ix)*idim*idim+(iy)*idim+(iz);
    el=el+1;
    for (k=0; k<el; k++) irow[k]=glob_row;
    if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(A),SM_DESCRIPTOR_P(A)))!=0)
      fprintf(stderr,"From psb_c_dspins: %d\n",info);
  }

  return(info);

}
