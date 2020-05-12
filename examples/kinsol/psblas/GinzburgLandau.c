
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
 * Example (psblas):
 * ----------------------------------------------------------------------
 *  Run command line: mpirun -np N -machinefile machines GinzburgLandau < \
 * \ kinsol.inp
 *  where N is the number of processors, and kinsol.inp contains the general
 * setting for the example.
 *
 * ----------------------------------------------------------------------
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <psb_util_cbind.h>

 #include <kinsol/kinsol.h>             /* access to KINSOL func., consts.     */
 #include <nvector/nvector_psblas.h>    /* access to PSBLAS N_Vector           */
 #include <sunmatrix/sunmatrix_psblas.h>/* access to PSBLAS SUNMATRIX 					*/
 #include <sunlinsol/sunlinsol_psblas.h>/* access to PSBLAS SUNLinearSolver    */
 #include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype     */
 #include <sundials/sundials_math.h>    /* access to SUNMAX, SUNRabs, SUNRsqrt */
 #include <sundials/sundials_iterative.h>


 #include <mpi.h>

 #define nb 20

 /* ------------------------------------------------
 Auxiliary functions for KINSOL
 -------------------------------------------------*/
 static int funcprpr(N_Vector u, N_Vector fval, void *user_data);
 static int jac(N_Vector y, N_Vector f, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2);
 static int check_flag(void *flagvalue, const char *funcname, int opt, int id);
 static void PrintFinalStats(void *kmem);

struct user_data_for_f {
  SUNMatrix *A;
  N_Vector *f;
  psb_i_t sizes[2];
  psb_d_t sqdeltah;
  psb_d_t epsilon;
};

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
 psb_d_t d(psb_d_t x, psb_d_t y, psb_d_t epsilon);
 psb_d_t g(psb_d_t x, psb_d_t y);
 psb_d_t f(psb_d_t x, psb_d_t y, psb_d_t epsilon);
 psb_d_t solution(psb_d_t x, psb_d_t y, psb_d_t epsilon);
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
   psb_c_SolverOptions options;             /* Solver options            */
   void *kmem;                              /* Pointer to KINSOL memory  */
   struct user_data_for_f user_data;        /* User data for computing F */
   /* Problem datas */
   N_Vector     u,fvec,constraints,sc,err,utrue;
   SUNMatrix    LAP,J;
   psb_d_t      deltah,deltah2,sqdeltah,x,y,zt[nb],ut[nb],errorvalue;
   psb_d_t      *val, epsilon, *valj;
   psb_l_t      *irow,*icol,*localvecindex;
   psb_i_t      icoeff,ilocalvec;
   /* Auxiliary variable for the construction of the communicator */
   psb_l_t     n,m,nt,nr,glob_row;
   psb_i_t     nlr,ijk[2],ijktemp[2],sizes[2],i,j,ii,ib,k;
   psb_l_t     *myidx;
   bool        owned;
   /* Flags & Timings*/
   psb_i_t     info;
   double      t1,t2;
   /* Input parameters */
   char methd[20],ptype[20];                /* Solve method and type     */
   char afmt[8],dump[1];
   psb_i_t nparms,dumpflag;
   psb_i_t idim,istop,itmax,itrace,irst,newtonmaxit;
   double          tol;                     /* tolerance for LS solution  */
   double  fnormtol, scsteptol;

   int globalstrategy = KIN_NONE;     /* Set global strategy flag */

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
     get_iparm(stdin,&newtonmaxit);
     get_dparm(stdin,&fnormtol);
     get_dparm(stdin,&scsteptol);
     get_hparm(stdin,dump);
     dumpflag = strcmp(dump,"T");
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
   psb_c_ibcast(ictxt,1,&newtonmaxit,0);
   psb_c_dbcast(ictxt,1,&fnormtol,0);
   psb_c_dbcast(ictxt,1,&scsteptol,0);
   psb_c_ibcast(ictxt,1,&dumpflag,0);

   if(iam == 0){
     printf("\n\nWelcome to the GinzburgLandau Test Program for KINSOL-PSBLAS\n\n");
   }

   epsilon = (psb_d_t) 1.0;

   /*-------------------------------------------------------------------------*
    * Information on the test Problem                                         *
    *-------------------------------------------------------------------------*/
    if(iam == 0){
      printf("Solver:\t%s\n",methd);
      printf("Precondiitioner:\t%s\n",ptype);
      printf("A format:\t\t%s\n",afmt);
      printf("Problem size:\t%d x %d\n",idim,idim);
      printf("The problem is running on %d processors\n",np);
      printf("\n\n");
    }



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

   /* The function F(u) = 0 can be written as:                                 *
    *   F(u) = -epsilon*Lap*u - u + u^3 - f = 0                                *
    * therefore we build just one time the matrix -epsilon*Lap and f, we reuse *
    * for every F(u)  computation. On the othere hand the non lineary is given *
    * by the function -u + u^3 and we will need to compute it every time on    *
    * given vector u.                                                          */
   LAP = NULL;
   LAP = SUNPSBLASMatrix(ictxt, cdh);
   if (LAP == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new matrix (LAP) \n\n");
       psb_c_abort(ictxt);
       return(1);
   }
   J = NULL;
   J = SUNPSBLASMatrix(ictxt, cdh);
   if (J == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new matrix (J) \n\n");
       SUNMatDestroy(LAP);
       psb_c_abort(ictxt);
       return(1);
   }
   fvec = NULL;
   fvec = N_VNew_PSBLAS(ictxt, cdh);
   if (fvec == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (fvec) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       psb_c_abort(ictxt);
       return(1);
   }
   constraints = NULL;
   constraints = N_VNew_PSBLAS(ictxt, cdh);
   if (constraints == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (constraints) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       N_VDestroy(fvec);
       psb_c_abort(ictxt);
       return(1);
   }
   u = NULL;
   u = N_VNew_PSBLAS(ictxt, cdh);
   if (constraints == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (u) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       N_VDestroy(fvec);
       N_VDestroy(constraints);
       psb_c_abort(ictxt);
       return(1);
   }
   sc = NULL;
   sc = N_VNew_PSBLAS(ictxt, cdh);
   if (sc == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (sc) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       N_VDestroy(fvec);
       N_VDestroy(u);
       N_VDestroy(constraints);
       psb_c_abort(ictxt);
       return(1);
   }
   err = NULL;
   err = N_VNew_PSBLAS(ictxt, cdh);
   if (err == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (err) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       N_VDestroy(fvec);
       N_VDestroy(u);
       N_VDestroy(constraints);
       N_VDestroy(sc);
       psb_c_abort(ictxt);
       return(1);
   }
   utrue = NULL;
   utrue = N_VNew_PSBLAS(ictxt, cdh);
   if (utrue == NULL) {
     if (iam == 0) printf("FAIL: Unable to create a new vector (utrue) \n\n");
       SUNMatDestroy(LAP);
       SUNMatDestroy(J);
       N_VDestroy(fvec);
       N_VDestroy(u);
       N_VDestroy(constraints);
       N_VDestroy(sc);
       N_VDestroy(utrue);
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
     localvecindex = (psb_l_t *) malloc( sizeof(psb_l_t)*ib );
     valj = (psb_d_t *) malloc( sizeof(psb_d_t)*ib );
     icoeff = 0;
     for(int kk = 0; kk < ib; kk++){
       i = ii + kk;
       glob_row = myidx[i]-1;              // Local Matrix Pointer
       psb_c_l_idx2ijk(ijk,glob_row,sizes,modes,base);
       x = (ijk[0]+1)*deltah;
       y = (ijk[1]+1)*deltah;

       zt[kk] = f(x,y,epsilon);
       ut[kk] = solution(x,y,epsilon);
       /* Internal point: build discretization                  */
       // term depending on   (x-1,y)
       val[icoeff] = -d(x,y,epsilon)/sqdeltah;
       if (ijk[0] == 0) {
         zt[kk] = g( (psb_d_t) 0, y )*(-val[icoeff]) + zt[kk];
       }else{
        ijktemp[0]   = ijk[0]-1;
        ijktemp[1]   = ijk[1];
        icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      }
      // term depending on     (x,y-1)
      val[icoeff] = -d(x,y,epsilon)/sqdeltah;
      if (ijk[1] == 0) {
        zt[kk] = g(x,(psb_d_t) 0)*(-val[icoeff]) + zt[kk];
        }else{
          ijktemp[0]   = ijk[0];
          ijktemp[1]   = ijk[1]-1;
          icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
          irow[icoeff] = glob_row;
          icoeff       = icoeff + 1;
      }
      // term depending on     (x,y)
        val[icoeff]  = ( (psb_d_t) 2.0 )*(d(x,y,epsilon) + d(x,y,epsilon))/sqdeltah;
        icol[icoeff] = psb_c_l_ijk2idx(ijk,sizes,modes,base);
        irow[icoeff] = glob_row;
        icoeff       = icoeff + 1;
      //  term depending on     (x,y+1)
        val[icoeff] = -d(x,y,epsilon)/sqdeltah;
        if(ijk[1] == idim - 1){
          zt[kk] = g(x,(psb_d_t) 1.0)*(-val[icoeff]) + zt[kk];
        }else{
          ijktemp[0]   = ijk[0];
          ijktemp[1]   = ijk[1]+1;
          icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
          irow[icoeff] = glob_row;
          icoeff       = icoeff + 1;
        }
      // term depending on     (x+1,y)
        val[icoeff] = -d(x,y,epsilon)/sqdeltah;
        if(ijk[0] == idim - 1){
          zt[kk] = g((psb_d_t) 1.0,y)*(-val[icoeff]) + zt[kk];
        }else{
          ijktemp[0]   = ijk[0]+1;
          ijktemp[1]   = ijk[1];
          icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
          irow[icoeff] = glob_row;
          icoeff       = icoeff + 1;
        }
      }
      info = SUNMatIns_PSBLAS(icoeff,irow,icol,val,LAP);
      info = SUNMatIns_PSBLAS(icoeff,irow,icol,val,J);
      if(info != 0) printf("%d Error in SUNMatIns! %d\n",iam,info);
      icoeff = 0;
      for(int jj=ii; jj < ii+ib; jj++){
        localvecindex[icoeff] = myidx[jj]-1;
        valj[icoeff] = 1.0;
        icoeff++;
      }
      info = psb_c_dgeins(ib,localvecindex,zt,NV_PVEC_P(fvec),NV_DESCRIPTOR_P(fvec));
      if(info != 0) printf("%d Error in dgeins! %d\n",iam,info);
      info = psb_c_dgeins(ib,localvecindex,ut,NV_PVEC_P(utrue),NV_DESCRIPTOR_P(utrue));
      if(info != 0) printf("%d Error in dgeins! %d\n",iam,info);
      if(info != 0) printf("%d Error in SUNMatIns! %d\n",iam,info);
      free(localvecindex);
    }
    /* Assembly of the various entities */
    psb_c_cdasb(cdh);
    N_VConst(0.0,constraints);
    N_VConst(1.0,u);
    N_VConst(1.0/sqdeltah,sc);
    N_VAsb_PSBLAS(fvec);
    SUNMatAsb_PSBLAS(J);
    SUNMatAsb_PSBLAS(LAP);

    /* We put the precomputed parts in the auxiliary data structure, this will
       be used to make both nonlinear function evaluations and Jacobian
       evaluations */
    user_data.A = &LAP;
    user_data.f = &fvec;
    user_data.sizes[0] = sizes[0];
    user_data.sizes[1] = sizes[1];
    user_data.sqdeltah = sqdeltah;
    user_data.epsilon  = epsilon;

    /* Call KINCreate/KINInit to initialize KINSOL:
       nvSpec is the nvSpec pointer used in the parallel version
       A pointer to KINSOL problem memory is returned and stored in kmem. */
    kmem = KINCreate();
    info = KINInit(kmem, funcprpr, u);
    if (check_flag(&info, "KINInit", 1, iam)) psb_c_abort(ictxt);
    info = KINSetNumMaxIters(kmem, newtonmaxit);
    if (check_flag(&info, "KINSetNumMaxIters", 1, iam)) psb_c_abort(ictxt);
    info = KINSetPrintLevel(kmem, 0);
    if (check_flag(&info, "KINSetPrintLevel", 1, iam)) psb_c_abort(ictxt);
    info = KINSetUserData(kmem, &user_data);
    if (check_flag(&info, "KINSetUserData", 1, iam)) psb_c_abort(ictxt);
    info = KINSetConstraints(kmem, constraints);
    if (check_flag(&info, "KINSetConstraints", 1, iam)) psb_c_abort(ictxt);
    info = KINSetFuncNormTol(kmem, fnormtol);
    if (check_flag(&info, "KINSetFuncNormTol", 1, iam)) psb_c_abort(ictxt);
    info = KINSetScaledStepTol(kmem, scsteptol);
    if (check_flag(&info, "KINSetScaledStepTol", 1, iam)) psb_c_abort(ictxt);

    /* We no longer need the constraints vector since KINSetConstraints
       creates a private copy for KINSOL to use. */
    N_VDestroy(constraints);

    /* We create now the linear systes solver */
    psb_c_DefaultSolverOptions(&options);
    options.eps    = tol;
    options.itmax  = itmax;
    options.irst   = irst;
    options.itrace = 1;
    options.istop  = istop;
    /* Create PSBLAS/MLD2P4 linear solver */
    LS = SUNLinSol_PSBLAS(options, methd, ptype, ictxt);
    if(check_flag((void *)LS, "SUNLinSol_PSBLAS", 0, iam)) psb_c_abort(ictxt);

    SUNLinSolInitialize_PSBLAS(LS);
    info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_SWEEPS",2);
    if (check_flag(&info, "SMOOTHER_SWEEPS", 1, iam)) psb_c_abort(ictxt);
    info = SUNLinSolSeti_PSBLAS(LS,"SUB_FILLIN",1);
    if (check_flag(&info, "SUB_FILLIN", 1, iam)) psb_c_abort(ictxt);
    info = SUNLinSolSetc_PSBLAS(LS,"COARSE_SOLVE","BJAC");
    if (check_flag(&info, "COARSE_SOLVE", 1, iam)) psb_c_abort(ictxt);
    info = SUNLinSolSetc_PSBLAS(LS,"COARSE_SUBSOLVE","ILU");
    if (check_flag(&info, "COARSE_SUBSOLVE", 1, iam)) psb_c_abort(ictxt);

    /* Attach the linear solver to KINSOL and set its options */
    info = KINSetLinearSolver(kmem, LS, J);
    if (check_flag(&info, "KINSetLinearSolver", 1, iam)) psb_c_abort(ictxt);
    info = KINSetJacFn(kmem,jac);
    if (check_flag(&info, "KINSetJacFn", 1, iam)) psb_c_abort(ictxt);
    info = KINSetEtaForm(kmem,KIN_ETACONSTANT);
    if (check_flag(&info, "KINSetEtaForm", 1, iam)) psb_c_abort(ictxt);
    info = KINSetEtaConstValue(kmem,options.eps);
    if (check_flag(&info, "KINSetEtaConstValue", 1, iam)) psb_c_abort(ictxt);


    /* Call KINSol and print output concentration profile */
    info = KINSol(kmem,           /* KINSol memory block */
                  u,              /* initial guess on input; solution vector */
                  globalstrategy, /* global strategy choice */
                  sc,    /* scaling vector for the variable u */
                  sc);            /* scaling vector for function values fval */

    if (check_flag(&info, "KINSol", 1, iam)) psb_c_abort(ictxt);

    N_VLinearSum((psb_d_t) 1.0, u, (psb_d_t) -1.0, utrue, err);
    errorvalue = psb_c_dgenrm2(NV_PVEC_P(err),NV_DESCRIPTOR_P(err));
    if(iam == 0) printf("\n\nProgram terminated with error %1.3e\n\n",errorvalue);

    if(iam == 0) PrintFinalStats(kmem);

    /* Print solution */
    if(dumpflag == 0){
      FILE *fid;
      char filename[20];
      sprintf(filename,"solution-%d.dat",iam);
      fid = fopen(filename,"w+");
      for(int i = 0; i < nlr; i++)
        fprintf(fid,"%1.16f\n",N_VGetArrayPointer_PSBLAS(u)[i]);
      fclose(fid);
    }
    /* Free the Memory */
    KINFree(&kmem);
    N_VDestroy(u);
    N_VDestroy(fvec);
    N_VDestroy(sc);
    N_VDestroy(err);
    N_VDestroy(utrue);
    SUNMatDestroy(LAP);
    SUNMatDestroy(J);
    SUNLinSolFree(LS);

    free(cdh);
    psb_c_exit(ictxt);

    return(0);
}

/*
*--------------------------------------------------------------------
* FUNCTIONS CALLED BY KINSOL
*--------------------------------------------------------------------
*/
static int funcprpr(N_Vector u, N_Vector fval, void *user_data)
{

  struct user_data_for_f *input = user_data;
  N_Vector cc;
  cc = N_VNew_PSBLAS(NV_ICTXT_P(u),NV_DESCRIPTOR_P(u));
  N_VAsb_PSBLAS(cc);
  N_VAsb_PSBLAS(fval);
  N_VLinearSum( (psb_d_t) 1.0,u,(psb_d_t) 0.0,cc,cc);     // cc <- u
  // F(u) = -epsilon^2*A*u - u -f + u^3
  SUNMatMatvec_PSBLAS( *input->A, cc, fval );             // fval <- Lap cc
  N_VLinearSum_PSBLAS( (psb_d_t) -1.0, *input->f, (psb_d_t) 1.0, fval, fval); // fval = fval - f
  N_VLinearSum_PSBLAS( (psb_d_t) -1.0, cc, (psb_d_t) 1.0, fval, fval);        // fval = fval - cc
  N_VProd_PSBLAS(cc,u,cc);                       // cc^2 = cc.*u
  N_VProd_PSBLAS(cc,u,cc);                       // cc^3 = u.*cc^2
  N_VLinearSum_PSBLAS( (psb_d_t) +1.0, cc, (psb_d_t) 1.0, fval, fval);        // fval = fval - cc^3

  N_VDestroy(cc);

  return(0);
}

static int jac(N_Vector yvec, N_Vector fvec, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2)
{

  struct user_data_for_f *input = user_data;
  psb_l_t     *myidx,*irow,*icol,glob_row;
  psb_i_t      nlr,ib,icoeff,ijk[2],ijktemp[2],modes=2,base=0,i,iam,np,info;
  psb_d_t     *val,x,y, epsilon = input->epsilon, zt[nb];
  psb_d_t     sqdeltah = input->sqdeltah, deltah;
  psb_i_t     *sizes  =  input->sizes;
  psb_i_t     idim = sizes[0];
  bool        owned;
  deltah = sqrt(sqdeltah);

  // Who are we?
  psb_c_info(SM_ICTXT_P(J),&iam,&np);
  if(iam == 0){
    printf("\tBuilding a new Jacobian\n");
    printf("\tSize of the grid %d x %d\n",sizes[0],sizes[1]);
    printf("\tepsilon = %1.2e deltah = %1.2e\n",epsilon,sqdeltah);
  }
  SUNMatZero(J); // We put to zero the old Jacobian to reuse the structure

  nlr = psb_c_cd_get_local_rows(SM_DESCRIPTOR_P(J));
  owned = SUNTRUE;
  myidx = (psb_l_t *) malloc( sizeof(psb_l_t)*nlr );
  psb_c_cd_get_global_indices(myidx,nlr,owned,NV_DESCRIPTOR_P(yvec));

  /* we build an auxiliary matrix consisting of one row at a time; just a
  small matrix. might be extended to generate a bunch of rows per call. */
  val = (psb_d_t *) malloc( sizeof(psb_d_t)*20*nb );
  irow = (psb_l_t *) malloc( sizeof(psb_l_t)*20*nb );
  icol = (psb_l_t *) malloc( sizeof(psb_l_t)*20*nb );

  for(int ii = 0; ii < nlr; ii += nb){
    ib = fmin(nb,nlr-ii+1);
    icoeff = 0;
    for(int kk = 0; kk < ib; kk++){
      i = ii + kk;
      glob_row = myidx[i]-1;              // Local Matrix Pointer
      psb_c_l_idx2ijk(ijk,glob_row,sizes,modes,base);
      x = (ijk[0]+1)*deltah;
      y = (ijk[1]+1)*deltah;

      zt[kk] = f(x,y,epsilon);
      /* Internal point: build discretization                  */
      // term depending on   (x-1,y)
      val[icoeff] = -d(x,y,epsilon)/sqdeltah;
      if (ijk[0] == 0) {
        zt[kk] = g( (psb_d_t) 0, y )*(-val[icoeff]) + zt[kk];
      }else{
       ijktemp[0]   = ijk[0]-1;
       ijktemp[1]   = ijk[1];
       icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
       irow[icoeff] = glob_row;
       icoeff       = icoeff + 1;
     }
     // term depending on     (x,y-1)
     val[icoeff] = -d(x,y,epsilon)/sqdeltah;
     if (ijk[1] == 0) {
       zt[kk] = g(x,(psb_d_t) 0)*(-val[icoeff]) + zt[kk];
       }else{
         ijktemp[0]   = ijk[0];
         ijktemp[1]   = ijk[1]-1;
         icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
         irow[icoeff] = glob_row;
         icoeff       = icoeff + 1;
     }
     // term depending on     (x,y)
       val[icoeff]  = ( (psb_d_t) 2.0 )*(d(x,y,epsilon) + d(x,y,epsilon))/sqdeltah
                -((psb_d_t) 1.0) + ((psb_d_t) 3.0)*pow((N_VGetArrayPointer_PSBLAS(yvec))[i],2.0);
       icol[icoeff] = psb_c_l_ijk2idx(ijk,sizes,modes,base);
       irow[icoeff] = glob_row;
       icoeff       = icoeff + 1;
     //  term depending on     (x,y+1)
       val[icoeff] = -d(x,y,epsilon)/sqdeltah;
       if(ijk[1] == idim - 1){
         zt[kk] = g(x,(psb_d_t) 1.0)*(-val[icoeff]) + zt[kk];
       }else{
         ijktemp[0]   = ijk[0];
         ijktemp[1]   = ijk[1]+1;
         icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
         irow[icoeff] = glob_row;
         icoeff       = icoeff + 1;
       }
     // term depending on     (x+1,y)
       val[icoeff] = -d(x,y,epsilon)/sqdeltah;
       if(ijk[0] == idim - 1){
         zt[kk] = g((psb_d_t) 1.0,y)*(-val[icoeff]) + zt[kk];
       }else{
         ijktemp[0]   = ijk[0]+1;
         ijktemp[1]   = ijk[1];
         icol[icoeff] = psb_c_l_ijk2idx(ijktemp,sizes,modes,base);
         irow[icoeff] = glob_row;
         icoeff       = icoeff + 1;
       }
     }
     info = SUNMatIns_PSBLAS(icoeff,irow,icol,val,J);
     if(info != 0) printf("%d Error in SUNMatIns! %d\n",iam,info);
   }

   SUNMatAsb_PSBLAS(J);
   if(iam == 0){
     printf("\tBuilding phase completed\n");
   }

   /* Free the Memory for the Auxiliary Array */
   free(val);
   free(irow);
   free(icol);
   free(myidx);

  return(0);
}

/* Coefficient functions for the LAP assembly */
psb_d_t d(psb_d_t x, psb_d_t y, psb_d_t epsilon){
 return((psb_d_t) pow(epsilon,2));
}
psb_d_t g(psb_d_t x, psb_d_t y){
 return((psb_d_t) 0.0);
}
psb_d_t f(psb_d_t x, psb_d_t y, psb_d_t epsilon){
  psb_d_t E = exp(1.0);
  return((-2.0*pow(E,2.0/epsilon)*(1.0 + cosh(1.0/epsilon)
      - 2.0*cosh((1.0 - x)/epsilon) -
        2.0*cosh(x/epsilon) - 2.0*cosh((1.0 - y)/epsilon)
      + 3.0*cosh((x - y)/epsilon) -
        2.0*cosh(y/epsilon) + 3.0*cosh((-1.0 + x + y)/epsilon)) +
     4096.0*pow(sinh((-1.0 + x)/(2.0*epsilon)),3.0)*
            pow(sinh(x/(2.0*epsilon)),3.0)*
            pow(sinh((-1.0 + y)/(2.0*epsilon)),3.0)*
            pow(sinh(y/(2.0*epsilon)),3.0))/pow(E,3.0/epsilon));
}
psb_d_t solution(psb_d_t x, psb_d_t y, psb_d_t epsilon){
  psb_d_t E = exp(1.0);
  return((16.0*sinh((-1.0 + x)/(2.0*epsilon))*sinh(x/(2.0*epsilon))*
    sinh((-1.0 + y)/(2.0*epsilon))*
    sinh(y/(2.0*epsilon)))/pow(E,1.0/epsilon));
}


/* KINSOL FLAG AND OUTPUT ROUTINES */

static int check_flag(void *flagvalue, const char *funcname, int opt, int id)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr,
            "\nSUNDIALS_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1);
  }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr,
              "\nSUNDIALS_ERROR(%d): %s() failed with flag = %d\n\n",
	      id, funcname, *errflag);
      return(1);
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr,
            "\nMEMORY_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1);
  }

  return(0);
}

static void PrintFinalStats(void *kmem)
{
  long int nni, nfe, nli, npe, nps, ncfl, nfeSG;
  psb_d_t funcnorm;
  int flag;

  flag = KINGetNumNonlinSolvIters(kmem, &nni);
  check_flag(&flag, "KINGetNumNonlinSolvIters", 1, 0);
  flag = KINGetNumFuncEvals(kmem, &nfe);
  check_flag(&flag, "KINGetNumFuncEvals", 1, 0);
  flag = KINGetNumLinIters(kmem, &nli);
  check_flag(&flag, "KINGetNumLinIters", 1, 0);
  flag = KINGetNumPrecEvals(kmem, &npe);
  check_flag(&flag, "KINGetNumPrecEvals", 1, 0);
  flag = KINGetNumPrecSolves(kmem, &nps);
  check_flag(&flag, "KINGetNumPrecSolves", 1, 0);
  flag = KINGetNumLinConvFails(kmem, &ncfl);
  check_flag(&flag, "KINGetNumLinConvFails", 1, 0);
  flag = KINGetNumLinFuncEvals(kmem, &nfeSG);
  check_flag(&flag, "KINGetNumLinFuncEvals", 1, 0);
  flag = KINGetFuncNorm(kmem, &funcnorm);
  check_flag(&flag, "KINGetFuncNorm", 1, 0);

  printf("\n\nFinal Statistics\n");
  printf("nni    = %5ld    nli   = %5ld\n", nni, nli);
  printf("nfe    = %5ld    nfeSG = %5ld\n", nfe, nfeSG);
  printf("nps    = %5ld    npe   = %5ld     ncfl  = %5ld\n", nps, npe, ncfl);
}
