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
 * Example (parallel):
 *
 * This example solves a nonlinear system that arises from a system
 * of partial differential equations. The PDE system is a food web
 * population model, with predator-prey interaction and diffusion on
 * the unit square in two dimensions. The dependent variable vector
 * is the following:
 *
 *       1   2         ns
 * c = (c , c ,  ..., c  )     (denoted by the variable cc)
 *
 * and the PDE's are as follows:
 *
 *                    i       i
 *         0 = d(i)*(c     + c    )  +  f  (x,y,c)   (i=1,...,ns)
 *                    xx      yy         i
 *
 *   where
 *
 *                   i             ns         j
 *   f  (x,y,c)  =  c  * (b(i)  + sum a(i,j)*c )
 *    i                           j=1
 *
 * The number of species is ns = 2 * np, with the first np being
 * prey and the last np being predators. The number np is both
 * the number of prey and predator species. The coefficients a(i,j),
 * b(i), d(i) are:
 *
 *   a(i,i) = -AA   (all i)
 *   a(i,j) = -GG   (i <= np , j >  np)
 *   a(i,j) =  EE   (i >  np,  j <= np)
 *   b(i) = BB * (1 + alpha * x * y)   (i <= np)
 *   b(i) =-BB * (1 + alpha * x * y)   (i >  np)
 *   d(i) = DPREY   (i <= np)
 *   d(i) = DPRED   ( i > np)
 *
 * The various scalar parameters are set using define's or in
 * routine InitUserData.
 *
 * The boundary conditions are: normal derivative = 0, and the
 * initial guess is constant in x and y, but the final solution
 * is not.
 *
 * The PDEs are discretized by central differencing on an MX by
 * MY mesh.
 *
 * The nonlinear system is solved by KINSOL using the method
 * specified in the local variable globalstrat.
 *
 * The preconditioner matrix is a block-diagonal matrix based on
 * the partial derivatives of the interaction terms f only.
 * -----------------------------------------------------------------
 * References:
 *
 * 1. Peter N. Brown and Youcef Saad,
 *    Hybrid Krylov Methods for Nonlinear Systems of Equations
 *    LLNL report UCRL-97645, November 1987.
 *
 * 2. Peter N. Brown and Alan C. Hindmarsh,
 *    Reduced Storage Matrix Methods in Stiff ODE systems,
 *    Lawrence Livermore National Laboratory Report  UCRL-95088,
 *    Rev. 1, June 1987, and  Journal of Applied Mathematics and
 *    Computation, Vol. 31 (May 1989), pp. 40-91. (Presents a
 *    description of the time-dependent version of this test
 *    problem.)
 * ----------------------------------------------------------------------
 *  Run command line: mpirun -np N -machinefile machines kinFoodWeb_kry_p
 *  where N = NPEX * NPEY is the number of processors.
 * ----------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <kinsol/kinsol.h>             /* access to KINSOL func., consts.     */
#include <nvector/nvector_psblas.h>    /* access to PSBLAS N_Vector           */
#include <sunmatrix/sunmatrix_psblas.h>/* access to PSBLAS SUNMATRIX 					*/
#include <sunlinsol/sunlinsol_psblas.h>/* access to PSBLAS SUNLinearSolver    */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype     */
#include <sundials/sundials_math.h>    /* access to SUNMAX, SUNRabs, SUNRsqrt */
#include <sundials/sundials_iterative.h>
#include <psb_util_cbind.h>

#include <mpi.h>

#define nb 20

/* ------------------------------------------------
Auxiliary functions for KINSOL
-------------------------------------------------*/
static int funcprpr(N_Vector cc, N_Vector fval, void *user_data);
/*-------------------------------------------------
 * Routine to read input from file
 *------------------------------------------------*/
#define LINEBUFSIZE 1024
static char buffer[LINEBUFSIZE+1];
int get_buffer(FILE *fp)
{
  char *info;
  while(!feof(fp)) {
    info = fgets(buffer,LINEBUFSIZE,fp);
    if (buffer[0]!='%') break;
  }
}
void get_iparm(FILE *fp, int *val)
{
  get_buffer(fp);
  sscanf(buffer,"%d ",val);
}
void get_dparm(FILE *fp, double *val)
{
  get_buffer(fp);
  sscanf(buffer,"%lf ",val);
}
void get_hparm(FILE *fp, char *val)
{
  get_buffer(fp);
  sscanf(buffer,"%s ",val);
}
/*-------------------------------------------------
 * Functions for the coefficients of the problem
 *------------------------------------------------*/
psb_d_t d(psb_d_t x, psb_d_t y);
psb_d_t g(psb_d_t x, psb_d_t y);
/*-------------------------------------------------
 * Main program
 *------------------------------------------------*/
int main(int argc, char *argv[]){

  psb_i_t      ictxt;                      /* PSBLAS Context            */
  MPI_Comm     comm;											 /* MPI Comminicator          */
  psb_i_t      np, iam;                    /* Number of procs, proc id  */
  psb_c_descriptor *cdh;                   /* PSBLAS Descriptor         */
  const psb_i_t base = 0;                  /* We use base = 0 for C ind.*/
  const psb_i_t modes = 2;                 /* Number of dimensions      */
  SUNLinearSolver LS;                      /* linear solver object      */
  void *kmem;                              /* Pointer to KINSOL memory  */
  /* Problem datas */
  N_Vector     cc;
  SUNMatrix    LAP;
  psb_d_t      deltah,deltah2,sqdeltah,x,y,zt[nb];
  psb_d_t      *val;
  psb_l_t      *irow,*icol;
  psb_i_t      icoeff;
  /* Auxiliary variable for the construction of the communicator */
  psb_l_t     n,m,nt,nr,glob_row;
  psb_i_t     nlr,ijk[2],ijktemp[2],sizes[2],i,j,ii,ib,k;
  psb_l_t     *myidx;
  bool        owned;
  /* Flags */
  psb_i_t     info;
  /* Input parameters */
  char methd[20],ptype[20];                /* Solve method and type     */
  char afmt[8];
  psb_i_t nparms;
  psb_i_t idim,istop,itmax,itrace,irst;
  double          tol;                     /* tolerance for LS solution  */

  /* Get processor number and total number of processes */
  ictxt = psb_c_init();
  psb_c_info(ictxt,&iam,&np);

  /* Read the detail of the test from the input file    */
  comm = MPI_Comm_f2c(ictxt);

  /* Read input and decide test-problem */
  psb_c_barrier(ictxt);
  if (iam == 0) {
    get_iparm(stdin,&nparms);
    get_hparm(stdin,methd);
    get_hparm(stdin,ptype);
    get_hparm(stdin,afmt);
    get_iparm(stdin,&idim);
    get_iparm(stdin,&istop);
    get_iparm(stdin,&itmax);
    get_iparm(stdin,&itrace);
    get_iparm(stdin,&irst);
    get_dparm(stdin,&tol);
  }
  /* Now broadcast the values, and check they're OK */
  psb_c_ibcast(ictxt,1,&nparms,0);
  psb_c_hbcast(ictxt,methd,0);
  psb_c_hbcast(ictxt,ptype,0);
  psb_c_hbcast(ictxt,afmt,0);
  psb_c_ibcast(ictxt,1,&idim,0);
  psb_c_ibcast(ictxt,1,&istop,0);
  psb_c_ibcast(ictxt,1,&itmax,0);
  psb_c_ibcast(ictxt,1,&itrace,0);
  psb_c_ibcast(ictxt,1,&irst,0);
  psb_c_dbcast(ictxt,1,&tol,0);


  /* Initialize array descriptor and sparse matrix storage by using a simple
   * BLOCK DISTRIBUTION of the data */
  cdh = psb_c_new_descriptor();
  psb_c_set_index_base(0);
  m = ((psb_l_t) idim )*idim;
  n = m;
  nt = (m+np-1)/np;
  nr = fmax(0,fmin(nt,m-(iam*nt)));
  nt = nr;

  MPI_Allreduce(MPI_IN_PLACE,&nt,1,MPI_LONG,MPI_SUM,comm);
  if(nt != m){
    printf("%d Initialization error %ld %ld %ld\n",iam,nr,nt,m);
    psb_c_barrier(ictxt);
    psb_c_abort(ictxt);
    return(1);
  }

  if (info=psb_c_cdall_nl(nr, ictxt, cdh)!=0) {
    fprintf(stderr,"From cdall: %d\nBailing out\n",info);
    psb_c_abort(ictxt);
  }
  nlr = psb_c_cd_get_local_rows(cdh);

  /* The function F(cc) = 0 can be written as:                                *
   *   F(cc) = diag(d(i))*Lap*cc + f(x,y,cc)                                  *
   * therefore we build just one time the matrix diag(d(i))*Lap and we reuse  *
   * for every F(cc) computation. On the othere hand the non lineary is given *
   * by the function f(x,y,cc) and we will need to compute it every time on   *
   * given vector cc.                                                         */
  LAP = NULL;
  LAP = SUNPSBLASMatrix(ictxt, cdh);
  if (LAP == NULL) {
    if (iam == 0) printf("FAIL: Unable to create a new matrix \n\n");
      psb_c_abort(ictxt);
      return(1);
  }

  /* We loop over rows belonging to current process using our BLOCK data
   * distribution.                                                     */
  deltah   = ((psb_d_t) 1.0)/(idim+1);
  sqdeltah = deltah*deltah;
  deltah2  = ((psb_d_t) 2.0)*deltah;
  sizes[0] = idim;
  sizes[1] = idim;

  owned = SUNTRUE;
  myidx = (psb_l_t *) malloc( sizeof(psb_l_t)*nlr );
  psb_c_cd_get_global_indices(myidx,nlr,owned,cdh);

  /* we build an auxiliary matrix consisting of one row at a time; just a
  small matrix. might be extended to generate a bunch of rows per call. */
  val = (psb_d_t *) malloc( sizeof(psb_d_t)*20*nb );
  irow = (psb_l_t *) malloc( sizeof(psb_l_t)*20*nb );
  icol = (psb_l_t *) malloc( sizeof(psb_l_t)*20*nb );

  psb_c_barrier(ictxt);
  for(int ii = 0; ii < nlr; ii += nb){
    ib = fmin(nb,nlr-ii+1);
    icoeff = 0;
    for(int kk = 0; kk < ib; kk++){
      i = ii + kk;
      glob_row = myidx[i]-1;              // Local Matrix Pointer
      psb_c_l_idx2ijk(ijk,glob_row,sizes,modes,base);
      x = (ijk[0]-1)*deltah;
      y = (ijk[1]-1)*deltah;

      zt[k] = (psb_d_t) 0.0;
      /* Internal point: build discretization                  */
        // term depending on   (x-1,y)
      val[icoeff] = -d(x,y)/sqdeltah;
      if (ijk[0] == 0) {
        zt[k] = g( (psb_d_t) 0, y )*(-val[icoeff]) + zt[k];
      }else{
        ijktemp[0]   = ijk[0]-1;
        ijktemp[1]   = ijk[1];
        icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      }
        // term depending on     (x,y-1)
      val[icoeff] = -d(x,y)/sqdeltah;
      if (ijk[1] == 0) {
        zt[k] = g(x,(psb_d_t) 0)*(-val[icoeff]) + zt[k];
      }else{
        ijktemp[0]   = ijk[0];
        ijktemp[1]   = ijk[1]-1;
        icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      }
        // term depending on     (x,y)
      val[icoeff]  = ( (psb_d_t) 2.0 )*(d(x,y) + d(x,y))/sqdeltah;
      icol[icoeff] = psb_c_l_ijk2idx(ijk,sizes,modes,base);
      irow[icoeff] = glob_row;
      icoeff       = icoeff + 1;
        //  term depending on     (x,y+1)
      val[icoeff] = -d(x,y)/sqdeltah;
      if(ijk[1] == idim - 1){
        zt[k] = g(x,(psb_d_t) 1.0)*(-val[icoeff]) + zt[k];
      }else{
        ijktemp[0]   = ijk[0];
        ijktemp[1]   = ijk[1]+1;
        icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      }
        // term depending on     (x+1,y)
      val[icoeff] = -d(x,y)/sqdeltah;
      if(ijk[0] == idim - 1){
        zt[k] = g((psb_d_t) 1.0,y)*(-val[icoeff]) + zt[k];
      }else{
        ijktemp[0]   = ijk[0]+1;
        ijktemp[1]   = ijk[1];
        icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      }
    }
    info = SUNMatIns_PSBLAS(icoeff,irow,icol,val,LAP);
  }
  /* Assembly of the matrix */
  psb_c_cdasb(cdh);
  SUNMatAsb_PSBLAS(LAP);

  /* Debug: */
  SUNPSBLASMatrix_Print(LAP, "2D Laplacian", "LaplacianTest.mtx");

  /* We can free here the auxiliary data we used to build the matrix */
  free(irow);
  free(icol);
  free(val);


  /* Initialize to NULL to avoid memory errors*/
  kmem = NULL;
  cc = NULL;

  /* Initial data */
  cc = N_VNew_PSBLAS(ictxt,cdh);

  /* */

  /* Call KINCreate/KINInit to initialize KINSOL: */
  kmem = KINCreate();
  info = KINInit(kmem, funcprpr, cc);


  /* Free memory */
  N_VDestroy(cc);
  SUNLinSolFree(LS);
  SUNMatDestroy(LAP);
  KINFree(&kmem);
  if ((info=psb_c_cdfree(cdh))!=0) {
    fprintf(stderr,"From cdfree: %d\nBailing out\n",info);
    psb_c_abort(ictxt);
  }

  free(cdh);
  psb_c_exit(ictxt);

  return(0);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY KINSOL
 *--------------------------------------------------------------------
 */
 static int funcprpr(N_Vector cc, N_Vector fval, void *user_data)
 {
   return(0);
 }

/* Coefficient functions for the LAP assembly */
psb_d_t d(psb_d_t x, psb_d_t y){
  return((psb_d_t) 1.0);
}
psb_d_t g(psb_d_t x, psb_d_t y){
  return((psb_d_t) 0.0);
}
