
/* ----------------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * ----------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ----------------------------------------------------------------------
 * Example (richards):
 * ----------------------------------------------------------------------
 *  Run command line: mpirun -np N -machinefile machines GinzburgLandau < \
 * \ kinsol.inp
 *  where N is the number of processors, and kinsol.inp contains the general
 * setting for the example.
 *
 * ----------------------------------------------------------------------
 * The structure of the application is the following:
 * 1. Initialize parallel environment with psb_init
 * 2. Initialize index space with psb_cdall
 * 3. Loop over the topology of the discretization mesh and build the descriptor
 *    with psb_cdins
 * 4. Assemble the descriptor with psb_cdasb
 * 5. Allocate the sparse matrices and dense vectors with psb_spall and psb_geall
 * 6. Loop over the time steps using the KINSOL routines to solve the set of
 *    nonlinear equation at each step.
 * -----------------------------------------------------------------------
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <psb_util_cbind.h>

 #include <kinsol/kinsol.h>             /* access to KINSOL func., consts.    */
 #include <nvector/nvector_psblas.h>    /* access to PSBLAS N_Vector          */
 #include <sunmatrix/sunmatrix_psblas.h>/* access to PSBLAS SUNMATRIX 			  */
 #include <sunlinsol/sunlinsol_psblas.h>/* access to PSBLAS SUNLinearSolver   */
 #include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype    */
 #include <sundials/sundials_math.h>    /* access to SUNMAX, SUNRabs, SUNRsqrt*/
 #include <sundials/sundials_iterative.h>


 #include <mpi.h>

 /* ------------------------------------------------
  * Auxiliary functions for KINSOL
 -------------------------------------------------*/
 // These functions are declared static to avoid conflicts with things that
 // could be somewhere else in the KINSOL library. It works also without the
 // static declaration, but you never know.
 static int funcprpr(N_Vector u, N_Vector fval, void *user_data);
 static int jac(N_Vector y, N_Vector f, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2);
 static int check_flag(void *flagvalue, const char *funcname, int opt, int id);

 /*--------------------------------------------------
  * Coefficient functions for J and F
  *-------------------------------------------------*/
  static double Sfun(double p, double alpha, double beta, double thetas,
                      double thetar);
  static double Kfun(double p, double a, double gamma, double Ks);
  static double Sfunprime(double p, double alpha, double beta, double thetas,
                      double thetar);
  static double Kfunprime(double p, double a, double gamma, double Ks);
  static double sgn(double x);

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
 /*-----------------------------------------------
  * Auxiliary data structures
  *-----------------------------------------------*/
  struct user_data_for_f {
    psb_l_t *vl;          // Local portion of the global indexs
    psb_i_t idim;         // Number of dofs in one direction
    psb_i_t nl;           // Number of blocks in the distribution
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks; // Problem Parameters
    psb_d_t dt;
    N_Vector *oldpressure; // Old Pressure value for Euler Time-Stepping
  };

 #define NBMAX       20

 int main(int argc, char *argv[]){

    void *kmem;                              /* Pointer to KINSOL memory    */
    SUNLinearSolver LS;                      /* linear solver object        */
    psb_c_SolverOptions options;             /* Solver options              */
    struct user_data_for_f user_data;        /* User data for computing F,J */
    /* BLOCK data distribution */
    psb_l_t ng,nl,*vl;
    psb_i_t nb;
    psb_l_t ix, iy, iz, el, glob_row;
    psb_l_t irow[10*NBMAX], icol[10*NBMAX];
    /* Problem data */
    N_Vector     u,constraints,su,sc;
    SUNMatrix    J;
    /* Input from file */
    psb_i_t nparms, idim, Nt, newtonmaxit, istop, itmax, itrace, irst;
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, Tmax, tol;
    double  fnormtol, scsteptol;
    char methd[20],ptype[20],afmt[8]; /* Solve method, p. type, matrix format */
    /* Parallel Environment */
    psb_i_t ictxt,iam,np;
    psb_c_descriptor *cdh;
    /* Time Stepping */
    psb_d_t dt;
    /* Auxiliary variable */
    int i,k;
    /* Flags */
    psb_i_t info;
    bool verbose = SUNFALSE;
    /* Set global strategy flag */
    int globalstrategy = KIN_NONE;

    ictxt = psb_c_init();
    psb_c_info(ictxt,&iam,&np);
    if(verbose){
      fprintf(stdout,"Initialization of the Richards miniapp I'm %d of %d\n",iam,np);
      fflush(stdout);
    }else{
      if(iam == 0)
      fprintf(stdout,"Initialization of the Richards miniapp on %d processes\n"
        ,np);
      fflush(stdout);
    }

    /* Read Problem Settings from file */
    if (iam == 0) {
      get_iparm(stdin,&nparms);
      get_iparm(stdin,&idim);
      get_dparm(stdin,&thetas);
      get_dparm(stdin,&thetar);
      get_dparm(stdin,&alpha);
      get_dparm(stdin,&beta);
      get_dparm(stdin,&a);
      get_dparm(stdin,&gamma);
      get_dparm(stdin,&Ks);
      get_dparm(stdin,&Tmax);
      get_iparm(stdin,&Nt);
      get_iparm(stdin,&newtonmaxit);
      get_dparm(stdin,&fnormtol);
      get_dparm(stdin,&scsteptol);
      get_hparm(stdin,methd);
      get_hparm(stdin,ptype);
      get_hparm(stdin,afmt);
      get_iparm(stdin,&istop);
      get_iparm(stdin,&itmax);
      get_iparm(stdin,&itrace);
      get_iparm(stdin,&irst);
      get_dparm(stdin,&tol);
      fprintf(stdout, "\nModel Parameters:\n");
      fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetar);
      fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetas);
      fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
      fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
      fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
        alpha,beta,a,gamma);
      fflush(stdout);
   }
   psb_c_ibcast(ictxt,1,&nparms,0);
   psb_c_ibcast(ictxt,1,&idim,0);
   psb_c_dbcast(ictxt,1,&thetas,0);
   psb_c_dbcast(ictxt,1,&thetar,0);
   psb_c_dbcast(ictxt,1,&alpha,0);
   psb_c_dbcast(ictxt,1,&beta,0);
   psb_c_dbcast(ictxt,1,&a,0);
   psb_c_dbcast(ictxt,1,&gamma,0);
   psb_c_dbcast(ictxt,1,&Ks,0);
   psb_c_dbcast(ictxt,1,&Tmax,0);
   psb_c_ibcast(ictxt,1,&Nt,0);
   psb_c_ibcast(ictxt,1,&newtonmaxit,0);
   psb_c_dbcast(ictxt,1,&fnormtol,0);
   psb_c_dbcast(ictxt,1,&scsteptol,0);
   psb_c_hbcast(ictxt,methd,0);
   psb_c_hbcast(ictxt,ptype,0);
   psb_c_hbcast(ictxt,afmt,0);
   psb_c_ibcast(ictxt,1,&istop,0);
   psb_c_ibcast(ictxt,1,&itmax,0);
   psb_c_ibcast(ictxt,1,&itrace,0);
   psb_c_ibcast(ictxt,1,&irst,0);
   psb_c_dbcast(ictxt,1,&tol,0);

    /* Perform a 3D BLOCK data distribution */
    if(iam==0){
      fprintf(stdout, "\nStarting 3D BLOCK data distribution\n");
      fflush(stdout);
    }
    cdh = psb_c_new_descriptor();
    ng = ((psb_l_t) idim)*idim*idim;
    nb = (ng+np-1)/np;
    nl = nb;
    if ( (ng -iam*nb) < nl) nl = ng -iam*nb;
      fprintf(stderr,"%d: Input data %d %ld %d %ld\n",iam,idim,ng,nb, nl);
    if ((vl=malloc(nb*sizeof(psb_l_t)))==NULL) {
      fprintf(stderr,"On %d: malloc failure\n",iam);
      psb_c_abort(ictxt);
   }
   i = ((psb_l_t)iam) * nb;
   for (k=0; k<nl; k++)
   vl[k] = i+k;

   if ((info=psb_c_cdall_vl(nl,vl,ictxt,cdh))!=0) {
      fprintf(stderr,"From cdall: %d\nBailing out\n",info);
      psb_c_abort(ictxt);
   }
   /* We store into the user_data_for_f the owned indices, these are then
   used to compute the Jacobian and the function evaluations */
   user_data.vl     = vl;
   user_data.idim   = idim;
   user_data.nl     = nl;
   user_data.thetas = thetas;
   user_data.thetar = thetar;
   user_data.alpha  = alpha;
   user_data.beta   = beta;
   user_data.a      = a;
   user_data.gamma  = gamma;
   user_data.Ks     = Ks;

   /*We need to reuse the same communicator many times, namely every time we
   need to populate a new Jacobian. Therefore we use the psb_c_cdins routine
   to generate the distributed adjacency graph for our problem.              */
   for (i=0; i<nl;  i++) {
     glob_row=vl[i];
     el = 0;
     ix = glob_row/(idim*idim);
     iy = (glob_row-ix*idim*idim)/idim;
     iz = glob_row-ix*idim*idim-iy*idim;
     /*  term depending on   (i-1,j,k)        */
     if(ix != 0){
       icol[el]=(ix-1)*idim*idim+(iy)*idim+(iz);
       el=el+1;
     }
     /*  term depending on     (i,j-1,k)        */
     if (iy != 0){
       icol[el]=(ix)*idim*idim+(iy-1)*idim+(iz);
       el=el+1;
     }
     /* term depending on      (i,j,k-1)        */
     if (iz != 0){
       icol[el]=(ix)*idim*idim+(iy)*idim+(iz-1);
       el=el+1;
     }
     /* term depending on      (i,j,k)          */
     icol[el]=(ix)*idim*idim+(iy)*idim+(iz);
     el=el+1;
     /*  term depending on     (i+1,j,k)        */
     if (iz != idim-1) {
       icol[el]=(ix)*idim*idim+(iy)*idim+(iz+1);
       el=el+1;
     }
     /*  term depending on     (i,j+1,k)        */
     if (iy != idim-1){
       icol[el]=(ix)*idim*idim+(iy+1)*idim+(iz);
       el=el+1;
     }
     /* term depending on      (i,j,k+1)        */
     if (ix != idim-1){
       icol[el]=(ix+1)*idim*idim+(iy)*idim+(iz);
       el=el+1;
     }
     for (k=0; k<el; k++) irow[k]=glob_row;
     if ((info=psb_c_cdins(el,irow,icol,cdh))!=0)
      fprintf(stderr,"From psb_c_cdins: %d\n",info);
   }

   if ((info=psb_c_cdasb(cdh))!=0)  return(info);

   /*-------------------------------------------------------
    * Linear Solver Setup and construction
    *-------------------------------------------------------*/
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


   /*-------------------------------------------------------
    * Solution vector and auxiliary data
    *------------------------------------------------------*/
    constraints = NULL;
    constraints = N_VNew_PSBLAS(ictxt, cdh);
    u = NULL;
    u = N_VNew_PSBLAS(ictxt, cdh);
    sc = NULL;
    sc = N_VNew_PSBLAS(ictxt, cdh);
    su = NULL;
    su = N_VNew_PSBLAS(ictxt, cdh);
    J = NULL;
    J = SUNPSBLASMatrix(ictxt, cdh);
    user_data.oldpressure = &u;

    N_VConst(0.0,constraints);
    N_VConst(1.0,u);
    N_VConst(1.0,sc);
    N_VConst(1.0,su);

   /*-------------------------------------------------------
    * We can now initialize the time loop
    -------------------------------------------------------*/
   dt = Tmax/(Nt+1);
   user_data.dt = dt;
   /* Initialization of the nonlinear solver */
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
   /* Attach the linear solver to KINSOL and set its options */
   info = KINSetLinearSolver(kmem, LS, J);
   if (check_flag(&info, "KINSetLinearSolver", 1, iam)) psb_c_abort(ictxt);
   info = KINSetJacFn(kmem,jac);
   if (check_flag(&info, "KINSetJacFn", 1, iam)) psb_c_abort(ictxt);
   info = KINSetEtaForm(kmem,KIN_ETACONSTANT);
   if (check_flag(&info, "KINSetEtaForm", 1, iam)) psb_c_abort(ictxt);
   info = KINSetEtaConstValue(kmem,options.eps);
   if (check_flag(&info, "KINSetEtaConstValue", 1, iam)) psb_c_abort(ictxt);

   // info = KINSol(kmem,           /* KINSol memory block */
   //               u,              /* initial guess on input; solution vector */
   //               globalstrategy, /* global strategy choice */
   //               su,             /* scaling vector for the variable u */
   //               sc);            /* scaling vector for function values fval */


   for(i=1;i==Nt+1;i++){  // Main Time Loop

   }

   /* Free the Memory */
   KINFree(&kmem);
   N_VDestroy(u);
   N_VDestroy(constraints);
   SUNMatDestroy(J);
   SUNLinSolFree(LS);
   free(cdh);

   psb_c_barrier(ictxt);
   psb_c_exit(ictxt);

   return(info);
}

/*
*--------------------------------------------------------------------
* FUNCTIONS CALLED BY KINSOL
*--------------------------------------------------------------------
*/
static int funcprpr(N_Vector u, N_Vector fval, void *user_data)
{
/* This function returns the evaluation fval = \Phi(u;parameters) to march the
   Newton method.                                                             */
   struct user_data_for_f *input = user_data;
   psb_i_t iam, np, ictxt, idim, nl;
   psb_i_t info;
   /* Problem parameters */
   psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks;

   /* Load problem parameters */
   thetas = input->thetas;
   thetar = input->thetar;
   alpha  = input->alpha;
   beta   = input->beta;
   gamma  = input->gamma;
   a      = input->a;
   Ks     = input->Ks;

   info = 0;


  return(info);
}

static int jac(N_Vector yvec, N_Vector fvec, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2)
{
  /* This function returns the evaluation of the Jacobian of the system upon
  request of the Newton method.                                               */
  struct user_data_for_f *input = user_data;
  psb_i_t iam, np, ictxt, idim, nl;
  psb_l_t ix, iy, iz, el,glob_row;
  psb_i_t i, k, info,ret;
  double x, y, z, deltah, sqdeltah, deltah2;
  double val[10*NBMAX], zt[NBMAX];
  psb_l_t irow[10*NBMAX], icol[10*NBMAX];
  /* Problem parameters */
  psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks;

  /* Load problem parameters */
  thetas = input->thetas;
  thetar = input->thetar;
  alpha  = input->alpha;
  beta   = input->beta;
  gamma  = input->gamma;
  a      = input->a;
  Ks     = input->Ks;

  info = 0;
  // Who am I?
  psb_c_info(SM_ICTXT_P(J),&iam,&np);
  if(iam == 0){
    fprintf(stdout,"\tBuilding a new Jacobian\n");
    fflush(stdout);
  }
  SUNMatZero(J); // We put to zero the old Jacobian to reuse the structure
  idim = input->idim;
  nl = input->nl;

  deltah = (double) 1.0/(idim+1);
  sqdeltah = deltah*deltah;
  deltah2  = 2.0* deltah;
  psb_c_set_index_base(0);
  for (i=0; i<nl;  i++) {
    glob_row=input->vl[i];
    el=0;
    // We get the local indexes:
    ix = glob_row/(idim*idim);
    iy = (glob_row-ix*idim*idim)/idim;
    iz = glob_row-ix*idim*idim-iy*idim;
    /*  Internal point: Build Discretization   */
    /*  term depending on     (i-1,j,k)        */

    /*  term depending on     (i,j-1,k)        */

    /* term depending on      (i,j,k-1)        */

    /* term depending on      (i,j,k)          */


    /*  term depending on     (i+1,j,k)        */

    /*  term depending on     (i,j+1,k)        */

    /* term depending on      (i,j,k+1)        */

    /* Get the corresponding rows of the matrix */
    for (k=0; k<el; k++) irow[k]=glob_row;
    /* Insert the local portion into the Jacobian */
    if ((ret=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)
      fprintf(stderr,"From psb_c_dspins: %d\n",ret);
  }

  // Assemble and return
  if ((info=psb_c_cdasb(SM_DESCRIPTOR_P(J)))!=0)  return(info);
  if ((info=psb_c_dspasb(SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)  return(info);

  return(info);
}

static double Sfun(double p, double alpha, double beta, double thetas,
                    double thetar){

  double s = 0.0;

  s = alpha*(thetas-thetar)/(alpha + pow(SUNRabs(p),beta));

  return(s);
}
static double Kfun(double p, double a, double gamma, double Ks){

  double K = 0.0;

  K = Ks*a/(a + pow(SUNRabs(p),gamma));

  return(K);

}
static double Sfunprime(double p, double alpha, double beta, double thetas,
                    double thetar){

  double s = 0.0;

  s = -alpha*beta*pow(SUNRabs(p),beta-1)*sgn(p)*(thetas-thetar)
            /pow(alpha + pow(SUNRabs(p),beta),2);

  return(s);

}
static double Kfunprime(double p, double a, double gamma, double Ks){

  double K = 0.0;

  K = -Ks*a*gamma*pow(SUNRabs(p),gamma-1)*sgn(p)
          /pow(a + pow(SUNRabs(p),gamma),2);

  return(K);
}

static double sgn(double x){
  // Can this be done in a better way?
  return((x > 0) ? 1 : ((x < 0) ? -1 : 0));
}

/*--------------------------------------------------------
 * KINSOL FLAG AND OUTPUT ROUTINES
 *-------------------------------------------------------*/

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
