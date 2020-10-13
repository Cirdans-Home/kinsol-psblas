
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
 static void PrintFinalStats(void *kmem, int i);

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
  static double source(double x, double y, double z, double t);
  static double boundary(double x, double y, double z, double t, void *user_data);
  static double upstream(double pU, double pL, void *user_data);
  static double chi(double pU, double pL);

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
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, rho, phi, pr; // Problem Parameters
    psb_d_t xmax,ymax,L;  // Size of the box
    psb_d_t dt;
    N_Vector oldpressure; // Old Pressure value for Euler Time-Stepping
    psb_i_t timestep;     // Actual time-step
  };

 #define NBMAX       20

 int main(int argc, char *argv[]){

    void *kmem;                              /* Pointer to KINSOL memory    */
    SUNLinearSolver LS;                      /* linear solver object        */
    psb_c_SolverOptions options;             /* Solver options              */
    struct user_data_for_f user_data;        /* User data for computing F,J */
    /* BLOCK data distribution */
    psb_l_t ng,nl,*vl;
    psb_i_t nb,sizes[3],ijk[3],ijkinsert[3];
    psb_l_t ix, iy, iz, el, glob_row;
    psb_l_t irow[10*NBMAX], icol[10*NBMAX];
    /* Problem data */
    N_Vector     u,constraints,su,sc;
    SUNMatrix    J;
    /* Input from file */
    psb_i_t nparms, idim, Nt, newtonmaxit, istop, itmax, itrace, irst;
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, Tmax, tol, rho, phi, pr;
    psb_d_t xmax, ymax, L;
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
    /* Performance variables */
    psb_d_t tic, toc, timecdh;

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
      get_dparm(stdin,&xmax);
      get_dparm(stdin,&ymax);
      get_dparm(stdin,&L);
      get_dparm(stdin,&thetas);
      get_dparm(stdin,&thetar);
      get_dparm(stdin,&alpha);
      get_dparm(stdin,&beta);
      get_dparm(stdin,&a);
      get_dparm(stdin,&gamma);
      get_dparm(stdin,&Ks);
      get_dparm(stdin,&rho);
      get_dparm(stdin,&phi);
      get_dparm(stdin,&pr);
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
      fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
      fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
      fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
      fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
        alpha,beta,a,gamma);
      fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
      fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf]\n",xmax,ymax,L);
      fflush(stdout);
   }
   psb_c_ibcast(ictxt,1,&nparms,0);
   psb_c_ibcast(ictxt,1,&idim,0);
   psb_c_dbcast(ictxt,1,&xmax,0);
   psb_c_dbcast(ictxt,1,&ymax,0);
   psb_c_dbcast(ictxt,1,&L,0);
   psb_c_dbcast(ictxt,1,&thetas,0);
   psb_c_dbcast(ictxt,1,&thetar,0);
   psb_c_dbcast(ictxt,1,&alpha,0);
   psb_c_dbcast(ictxt,1,&beta,0);
   psb_c_dbcast(ictxt,1,&a,0);
   psb_c_dbcast(ictxt,1,&gamma,0);
   psb_c_dbcast(ictxt,1,&Ks,0);
   psb_c_dbcast(ictxt,1,&rho,0);
   psb_c_dbcast(ictxt,1,&phi,0);
   psb_c_dbcast(ictxt,1,&pr,0);
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

   if( (xmax != ymax) || (xmax != L) || (ymax != L)){
     fprintf(stderr, "\nAbort, works only on cube for now!\n");
     fflush(stderr);
     psb_c_abort(ictxt);
   }


    /* Perform a 3D BLOCK data distribution */
    if(iam==0){
      fprintf(stdout, "\nStarting 3D BLOCK data distribution\n");
      fflush(stdout);
    }
    psb_c_barrier(ictxt);
    cdh = psb_c_new_descriptor();
    psb_c_set_index_base(0);
    ng = ((psb_l_t) idim)*idim*idim;
    nb = (ng+np-1)/np;
    nl = nb;
    if ( (ng -iam*nb) < nl) nl = ng -iam*nb;
      fprintf(stdout,"%d: Input data %d %ld %d %ld\n",iam,idim,ng,nb, nl);
    if ((vl=malloc((nb+1)*sizeof(psb_l_t)))==NULL) {
      fprintf(stderr,"On %d: malloc failure\n",iam);
      psb_c_abort(ictxt);
   }
   i = ((psb_l_t)iam) * nb;
   for (k=0; k<= nl; k++){
    vl[k] = i+k;
   }
   if ((info=psb_c_cdall_vl(nl,vl,ictxt,cdh))!=0) {
      fprintf(stderr,"From cdall: %d\nBailing out\n",info);
      psb_c_abort(ictxt);
   }
   /* We store into the user_data_for_f the owned indices, these are then
   used to compute the Jacobian and the function evaluations */
   user_data.vl     = vl;
   user_data.idim   = idim;
   user_data.xmax   = xmax;
   user_data.ymax   = ymax;
   user_data.L      = L;
   user_data.nl     = nl;
   user_data.thetas = thetas;
   user_data.thetar = thetar;
   user_data.alpha  = alpha;
   user_data.beta   = beta;
   user_data.a      = a;
   user_data.gamma  = gamma;
   user_data.Ks     = Ks;
   user_data.rho    = rho;
   user_data.phi    = phi;
   user_data.pr     = pr;

   /*We need to reuse the same communicator many times, namely every time we
   need to populate a new Jacobian. Therefore we use the psb_c_cdins routine
   to generate the distributed adjacency graph for our problem.              */
   sizes[0] = idim; sizes[1] = idim; sizes[2] = idim;
   for (i=0; i <= nl;  i++) {
     glob_row=vl[i];
     el = 0;
     psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
     ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
     /*  term depending on   (i-1,j,k)        */
     if(ix != 0){
       ijkinsert[0]=ix-1; ijkinsert[1]=iy; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /*  term depending on     (i,j-1,k)        */
     if (iy != 0){
       ijkinsert[0]=ix; ijkinsert[1]=iy-1; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i,j,k-1)        */
     if (iz != 0){
       ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz-1;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i,j,k)          */
     ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz;
     icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
     el=el+1;
     /*  term depending on     (i+1,j,k)        */
     if (iz != idim-1) {
       ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz+1;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /*  term depending on     (i,j+1,k)        */
     if (iy != idim-1){
       ijkinsert[0]=ix-1; ijkinsert[1]=iy+1; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i,j,k+1)        */
     if (ix != idim-1){
       ijkinsert[0]=ix+1; ijkinsert[1]=iy; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     for (k=0; k<el; k++) irow[k]=glob_row;
     if ((info=psb_c_cdins(el,irow,icol,cdh))!=0)
      fprintf(stderr,"From psb_c_cdins: %d\n",info);
   }

   tic = psb_c_wtime();
   if ((info=psb_c_cdasb(cdh))!=0)  return(info);
   toc = psb_c_wtime();
   timecdh = toc-tic;


   if (iam == 0){
     printf("Built communicator on %ld global rows\n",psb_c_cd_get_global_rows(cdh));
     fprintf(stdout,"Communicator Building time: %lf s\n",timecdh);
   }

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

    if(iam == 0) fprintf(stdout, "Setting ML options.\n");
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
    user_data.oldpressure = N_VNew_PSBLAS(ictxt, cdh);

    N_VConst(0.0,constraints);      // No constraints
    N_VConst(1.0,sc);               // Unweighted norm
    N_VConst(1.0,su);               // Unweighted norm

   /*-------------------------------------------------------
    * We can now initialize the time loop
    -------------------------------------------------------*/
   dt = Tmax/(Nt+1);
   user_data.dt = dt;
   N_VConst(pr,u);  // Initial Condition
   N_VLinearSum(1.0,u,0.0,user_data.oldpressure,user_data.oldpressure);
   /* Initialization of the nonlinear solver */
   kmem = KINCreate();
   info = KINInit(kmem, funcprpr, u);
   if (check_flag(&info, "KINInit", 1, iam)) psb_c_abort(ictxt);
   info = KINSetNumMaxIters(kmem, newtonmaxit);
   if (check_flag(&info, "KINSetNumMaxIters", 1, iam)) psb_c_abort(ictxt);
   info = KINSetPrintLevel(kmem, 0);
   if (check_flag(&info, "KINSetPrintLevel", 2, iam)) psb_c_abort(ictxt);
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

   psb_c_barrier(ictxt);
   if (iam == 0){
     fprintf(stdout, "\n**********************************************************************\n");
     fprintf(stdout, "************ Time Step %d of %d ****************************************\n", 1,Nt );
   }
   info = KINSol(kmem,           /* KINSol memory block */
                 u,              /* initial guess on input; solution vector */
                 globalstrategy, /* global strategy choice */
                 su,             /* scaling vector for the variable u */
                 sc);            /* scaling vector for function values fval */


   if (check_flag(&info, "KINSol", 1, iam)){
     psb_c_abort(ictxt);
   } else {
     info = 0;
   }

   if(iam == 0){
     PrintFinalStats(kmem,1);
   }
   KINFree(&kmem);

   for(i=2;i<=Nt;i++){  // Main Time Loop
     if (iam == 0){
       fprintf(stdout, "\n**********************************************************************\n");
       fprintf(stdout, " Time Step %d of %d \n", i,Nt );
       fprintf(stdout, "**********************************************************************\n");
       fflush(stdout);
     }
     user_data.timestep = i; // used to compute time depending quantities

     /* For Euler Time-Stepping we take note of the old pressure value */
     N_VLinearSum(1.0,u,0.0,user_data.oldpressure,user_data.oldpressure);

     /* We perform the new incomplete Newton time step using as starting point
     the solution at the previous time step.                                  */
     N_VConst(1.0,sc);               // Unweighted norm
     N_VConst(1.0,su);               // Unweighted norm
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
     info = KINSol(kmem,           /* KINSol memory block */
                   u,              /* initial guess on input; solution vector */
                   globalstrategy, /* global strategy choice */
                   su,             /* scaling vector for the variable u */
                   sc);            /* scaling vector for function values fval */

     if (check_flag(&info, "KINSol", 1, iam)){
       psb_c_abort(ictxt);
     } else {
       info = 0;
     }

     if(iam == 0){
       PrintFinalStats(kmem,1);
     }
     KINFree(&kmem);
   }

   /* Free the Memory */

   N_VDestroy(u);
   N_VDestroy(constraints);
   N_VDestroy(sc);
   N_VDestroy(su);
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
/* This function returns the evaluation fval = Φ(u;parameters) to march the
   Newton method.                                                             */
   struct user_data_for_f *input = user_data;
   N_Vector uold;
   psb_i_t iam, np, ictxt, idim, nl;
   psb_i_t i, k, info;
   psb_l_t glob_row, irow[1];
   double x, y, z, t, deltah, sqdeltah, deltah2;
   double val[1],entries[8];
   psb_i_t ix, iy, iz, ijk[3],sizes[3];

   /* Problem parameters */
   psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, dt, rho, phi, pr;
   psb_d_t xmax, ymax, L;

   /* Load problem parameters */
   thetas = input->thetas;
   thetar = input->thetar;
   alpha  = input->alpha;
   beta   = input->beta;
   gamma  = input->gamma;
   a      = input->a;
   Ks     = input->Ks;
   rho    = input->rho;
   phi    = input->phi;
   uold   = input->oldpressure;
   dt     = input->dt;
   pr     = input->pr;
   xmax   = input->xmax;
   ymax   = input->ymax;
   L      = input->L;

   info = 0;

   idim = input->idim;
   nl = input->nl;

   // Who am I?
   psb_c_info(NV_ICTXT_P(u),&iam,&np);
   psb_c_set_index_base(0);

   if (iam == 0){
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fprintf(stdout, "Function Evaluation on the parameters:\n");
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetar);
     fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetas);
     fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
     fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
     fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
     fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
     fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
     alpha,beta,a,gamma);
     fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
     fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf]\n",xmax,ymax,L);
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fflush(stdout);
   }

   deltah = (double) L/(idim+1);
   sqdeltah = deltah*deltah;
   deltah2  = 2.0* deltah;
   sizes[0] = idim; sizes[1] = idim; sizes[2] = idim;

   for (i=0; i<nl;  i++) {
     glob_row=input->vl[i];                 // Get the index of the global row
     // We compute the local indexes of the elements on the stencil
     psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
     ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
     x = ix*deltah; y = iy*deltah; z = iz*deltah; t = (input->timestep)*dt;
     // We compute the result of Φ(p) by first going back to the (i,j,k)
     // indexing and substiting the value of p[i,j,k] on the boundary with the
     // correct values, otherwise we use the entries stored in u, together with
     // the halo values to produce the NVector fval = Φ(p). Another way of doing
     // this would be assembling every time a bunch of temporary matrices
     // with the values of the nonlinear evaluations and doing some
     // matrix-vector products. This way should be faster, and less taxing on
     // the memory.
     entries[0] = psb_c_dgetelem(NV_PVEC_P(uold),glob_row,
                                 NV_DESCRIPTOR_P(uold)); // u^(l-1)_{i,j,k}
     entries[1] = psb_c_dgetelem(NV_PVEC_P(u),glob_row,
                                  NV_DESCRIPTOR_P(u)); // u^(l)_{i,j,k}
     if (ix == 0) {        // Cannot do i-1
       entries[2] = boundary(x,y,z,t,user_data); // u^(l)_{i-1,j,k}
     }else{
       ijk[0] = ix - 1; ijk[1] = iy; ijk[2] = iz;
       entries[2] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u)); // u^(l)_{i-1,j,k}
     }
     if (ix == idim -1){
       entries[3] = boundary(x,y,z,t,user_data);
     }else{
       ijk[0] = ix+1; ijk[1] = iy; ijk[2] = iz;
       entries[3] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i+1,j,k}
     }
     if (iy == 0){       // Cannot do j-1
       entries[4] = boundary(x,y,z,t,user_data); // u^(l)_{i+1,j,k}
     }else{
       ijk[0] = ix; ijk[1] = iy-1; ijk[2] = iz;
       entries[4] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j-1,k}
     }
     if (iy == idim -1){
       entries[5] = boundary(x,y,z,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy+1; ijk[2] = iz;
       entries[5] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j+1,k}
     }
     if (iz == 0){       // Cannot do k-1
       entries[6] = boundary(x,y,z,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy; ijk[2] = iz-1;
       entries[6] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j,k-1}
     }
     if (iz == idim -1){ // Cannot do k+1
       entries[7] = boundary(x,y,L,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy; ijk[2] = iz+1;
       entries[7] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j,k+1}
     }
     // We have now recovered all the entries, and we can compute the glob_rowth
     // entry of the funciton
     val[0] = (rho*phi)/dt*(Sfun(entries[0],alpha,beta,thetas,thetar)
      - Sfun(entries[1],alpha,beta,thetas,thetar))
      - 1/sqdeltah*(  // x-direction
        ( entries[3]-entries[1] )*upstream(entries[3],entries[1],user_data)
         - (entries[1]-entries[2])*upstream(entries[1],entries[2],user_data)
      )
      - 1/sqdeltah*(  // y-direction
        ( entries[5]-entries[1] )*upstream(entries[5],entries[1],user_data)
         - (entries[1]-entries[4])*upstream(entries[1],entries[4],user_data)
      )
      - 1/sqdeltah*(  // z-direction
        ( entries[7]-entries[1] )*upstream(entries[7],entries[1],user_data)
         - (entries[1]-entries[6])*upstream(entries[1],entries[6],user_data)
      )
      - 1/deltah2*( // z-transport
        Kfun(entries[7],a,gamma,Ks)
        -Kfun(entries[6],a,gamma,Ks)
      )- source(x,y,z,t);

      // fprintf(stdout, "\n\nGlobal Row %ld Central index (%d,%d,%d)\n",glob_row,ix,iy,iz);
      // fprintf(stdout, "[ 00.00 , %1.2lf , 00.00 ]\n[%1.2lf , %1.2lf , %1.2lf ]\n[ 00.00 , %1.2lf , 00.00 ]\n",entries[5],entries[2],entries[1],entries[3],entries[4]);
      // fprintf(stdout, "[%1.2lf]\n[%1.2lf]\n[%1.2lf]\n",entries[7],entries[1],entries[6]);
      // fprintf(stdout, "F(%ld|%d,%d,%d) = %lf\n",glob_row,ix,iy,iz,val[0]);

     irow[0] = glob_row;
     psb_c_dgeins(1,irow,val,NV_PVEC_P(fval),NV_DESCRIPTOR_P(fval));
   }

   // We assemble the vector at the end
   N_VAsb_PSBLAS(fval);

   FILE *outfile;
   outfile = fopen("vector.dat","w+");
   N_VPrintFile_PSBLAS(fval,outfile);
   fclose(outfile);
  return(info);
}

static int jac(N_Vector yvec, N_Vector fvec, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2)
{
  /* This function returns the evaluation of the Jacobian of the system upon
  request of the Newton method.                                               */
  struct user_data_for_f *input = user_data;
  psb_i_t iam, np, ictxt, idim, nl;
  psb_i_t i, k, info, el;
  psb_l_t glob_row, irow[10*NBMAX], icol[10*NBMAX];
  double x, y, z, t, deltah, sqdeltah, deltah2;
  double val[10*NBMAX],entries[8];
  psb_i_t ix, iy, iz, ijk[3],ijkinsert[3],sizes[3];

  /* Problem parameters */
  psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, dt, rho, phi, pr;
  psb_d_t xmax, ymax, L;
  /* Timings */
  psb_d_t tic, toc, timecdh;

  /* Load problem parameters */
  thetas = input->thetas;
  thetar = input->thetar;
  alpha  = input->alpha;
  beta   = input->beta;
  gamma  = input->gamma;
  a      = input->a;
  Ks     = input->Ks;
  dt     = input->dt;
  rho    = input->rho;
  phi    = input->phi;
  pr     = input->pr;
  xmax   = input->xmax;
  ymax   = input->ymax;
  L      = input->L;

  info = 0;

  idim = input->idim;
  nl = input->nl;

  // Who am I?
  psb_c_info(NV_ICTXT_P(yvec),&iam,&np);
  psb_c_set_index_base(0);

  if (iam == 0){
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fprintf(stdout, "Jacobian Evaluation on the parameters:\n");
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetar);
    fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetas);
    fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
    fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
    fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
    fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
    fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
    alpha,beta,a,gamma);
    fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
    fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf]\n",xmax,ymax,L);
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fflush(stdout);
  }

  deltah = (double) L/(idim+1);
  sqdeltah = deltah*deltah;
  deltah2  = 2.0* deltah;
  sizes[0] = idim; sizes[1] = idim; sizes[2] = idim;
  x = ix*deltah; y = iy*deltah; z = iz*deltah; t = (input->timestep)*dt;

  for (i=0; i<nl;  i++) {
    glob_row=input->vl[i];                 // Get the index of the global row
    // We compute the local indexes of the elements on the stencil
    psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
    ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
    el = 0;

    /* To compute the expression of the Jacobian for Φ we first need to access
    the entries for the current iterate, these are contained in the N_Vector
    yvec. Another way of doing this would be assembling every time a bunch of
    temporary matrices with the values of the nonlinear evaluations and doing
    some matrix-matrix operationss. This way should be faster, and less taxing
    on the memory.                                                            */
    entries[1] = psb_c_dgetelem(NV_PVEC_P(yvec),glob_row,
                                 NV_DESCRIPTOR_P(yvec)); // u^(l)_{i,j,k}
    if (ix == 0) {        // Cannot do i-1
      entries[2] = boundary(x,y,z,t,&user_data); // u^(l)_{i-1,j,k}
    }else{
      ijk[0] = ix - 1; ijk[1] = iy; ijk[2] = iz;
      entries[2] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec)); // u^(l)_{i-1,j,k}
    }
    if (ix == idim -1){
      entries[3] = boundary(x,y,z,t,&user_data);
    }else{
      ijk[0] = ix+1; ijk[1] = iy; ijk[2] = iz;
      entries[3] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i+1,j,k}
    }
    if (iy == 0){       // Cannot do j-1
      entries[4] = boundary(x,y,z,t,&user_data); // u^(l)_{i+1,j,k}
    }else{
      ijk[0] = ix; ijk[1] = iy-1; ijk[2] = iz;
      entries[4] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j-1,k}
    }
    if (iy == idim -1){
      entries[5] = boundary(x,y,z,t,&user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy+1; ijk[2] = iz;
      entries[5] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j+1,k}
    }
    if (iz == 0){       // Cannot do k-1
      entries[6] = boundary(x,y,z,t,&user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy; ijk[2] = iz-1;
      entries[6] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k-1}
    }
    if (iz == idim -1){ // Cannot do k+1
      entries[7] = boundary(x,y,L,t,&user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy; ijk[2] = iz+1;
      entries[7] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k+1}
    }

    /* Now that we have all the entries of yvec needed to compute the entries
    of the current row of the Jacobian we can loop through the different nonzero
    elements and compute the relative values.                                 */
    /*  term depending on   (i-1,j,k)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[2],a,gamma,Ks)*chi(entries[1],entries[2])
      -entries[2]*Kfunprime(entries[2],a,gamma,Ks)*chi(entries[1],entries[2])
      -1*upstream(entries[1],entries[2],user_data)
    );
    if(ix != 0){
      ijkinsert[0]=ix-1; ijkinsert[1]=iy; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    /*  term depending on     (i,j-1,k)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[4],a,gamma,Ks)*chi(entries[1],entries[4])
      -entries[2]*Kfunprime(entries[4],a,gamma,Ks)*chi(entries[1],entries[4])
      -1*upstream(entries[1],entries[4],user_data)
    );
    if (iy != 0){
      ijkinsert[0]=ix; ijkinsert[1]=iy-1; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    /* term depending on      (i,j,k-1)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[6],a,gamma,Ks)*chi(entries[1],entries[6])
      -entries[2]*Kfunprime(entries[6],a,gamma,Ks)*chi(entries[1],entries[6])
      -1*upstream(entries[1],entries[6],user_data)
    ) + Kfunprime(entries[6],a,gamma,Ks)/deltah2;
    if (iz != 0){
      ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz-1;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    /* term depending on      (i,j,k)          */
    val[el] = rho*phi*Sfunprime(entries[1],alpha,beta,thetas,thetar)/dt
      -(entries[2]*Kfunprime(entries[1],a,gamma,Ks)*chi(entries[1],entries[2]))/sqdeltah
      -(entries[4]*Kfunprime(entries[1],a,gamma,Ks)*chi(entries[1],entries[4]))/sqdeltah
      -(entries[6]*Kfunprime(entries[1],a,gamma,Ks)*chi(entries[1],entries[6]))/sqdeltah
      -(entries[3]*Kfunprime(entries[1],a,gamma,Ks)*(1.0-chi(entries[3],entries[1])))/sqdeltah
      -(entries[5]*Kfunprime(entries[1],a,gamma,Ks)*(1.0-chi(entries[5],entries[1])))/sqdeltah
      -(entries[7]*Kfunprime(entries[1],a,gamma,Ks)*(1.0-chi(entries[7],entries[1])))/sqdeltah
      +(entries[1]*Kfunprime(entries[1],a,gamma,Ks)/sqdeltah)*(
        (1.0-chi(entries[3],entries[1]))+chi(entries[1],entries[2])
        +(1.0-chi(entries[5],entries[1]))+chi(entries[1],entries[4])
        +(1.0-chi(entries[7],entries[1]))+chi(entries[1],entries[6])
      )
      +(1/sqdeltah)*(
        upstream(entries[3],entries[1],user_data)
        +upstream(entries[1],entries[2],user_data)
        +upstream(entries[5],entries[1],user_data)
        +upstream(entries[1],entries[6],user_data)
        +upstream(entries[7],entries[1],user_data)
        +upstream(entries[1],entries[6],user_data)
      );
    ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz;
    icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
    el=el+1;
    //fprintf(stdout, "val = %e\n",val[el]);
    /*  term depending on     (i+1,j,k)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[3],a,gamma,Ks)*chi(entries[3],entries[1])
      - entries[3]*Kfunprime(entries[3],a,gamma,Ks)*chi(entries[3],entries[1])
      - upstream(entries[3],entries[1],user_data)
    );
    if (iz != idim-1) {
      ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz+1;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    /*  term depending on     (i,j+1,k)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[5],a,gamma,Ks)*chi(entries[5],entries[1])
      - entries[5]*Kfunprime(entries[5],a,gamma,Ks)*chi(entries[5],entries[1])
      - upstream(entries[5],entries[1],user_data)
    );
    if (iy != idim-1){
      ijkinsert[0]=ix-1; ijkinsert[1]=iy+1; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    /* term depending on      (i,j,k+1)        */
    val[el] = 1/sqdeltah*(
      entries[1]*Kfunprime(entries[7],a,gamma,Ks)*chi(entries[7],entries[1])
      - entries[7]*Kfunprime(entries[7],a,gamma,Ks)*chi(entries[7],entries[1])
      - upstream(entries[7],entries[1],user_data)
    ) - Kfunprime(entries[7],a,gamma,Ks)/deltah2;
    if (ix != idim-1){
      ijkinsert[0]=ix+1; ijkinsert[1]=iy; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    //fprintf(stdout, "val = %e\n",val[el]);
    for (k=0; k<el; k++) irow[k]=glob_row;
    /* Insert the local portion into the Jacobian */
    if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)
     fprintf(stderr,"From psb_c_dspins: %d\n",info);
  }

  // Assemble and return
  // if ((info=psb_c_cdasb(SM_DESCRIPTOR_P(J)))!=0)  return(info);
  tic = psb_c_wtime();
  if ((info=psb_c_dspasb(SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)  return(info);
  toc = psb_c_wtime();
  if (iam == 0) fprintf(stdout, "Buit new Jacobian in %lf s\n",toc-tic);

  SUNPSBLASMatrix_Print(J,"Jacobian","Jacobian.mtx");

  return(info);
}

static double Sfun(double p, double alpha, double beta, double thetas,
                    double thetar){

  double s = 0.0;

  s = alpha*(thetas-thetar)/(alpha + pow(SUNRabs(p),beta)) + thetar;

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

static double source(double x, double y, double z, double t){
  // Source term function
  return(0.0);
}

static double boundary(double x, double y, double z, double t, void *user_data){
  /* Dirichlet boundary, (possibly) time dependent */
  /* Water is applied at z=L such that the pressure head becomes zero
     in the region, and p = pr on the all the remaining boundaries.*/
  struct user_data_for_f *input = user_data;
  psb_d_t pr, xmax, ymax, L;
  psb_d_t res;

  pr     = input->pr;
  xmax   = input->xmax;
  ymax   = input->ymax;
  L      = input->L;

  if (z == L){
    if( x >= xmax/4.0 & x <= 3.0*xmax/4.0 & y >= ymax/4.0 & y <= 3.0*ymax/4.0){
      res = 0.0;
    }
    else{
      res = pr;
    }
  }
  else{
    res = pr;
  }
  return(res);
}

static double upstream(double pU, double pL, void *user_data){
  /* Upstream mean for the Ks function */
  struct user_data_for_f *input = user_data;
  double alpha, gamma, Ks;
  double res;

  alpha  = input->alpha;
  gamma  = input->gamma;
  Ks     = input->Ks;

  if(pU-pL >= 0){
    res = Kfun(pU,alpha,gamma,Ks);
  }else{
    res = Kfun(pL,alpha,gamma,Ks);
  }
  return(res);
}

static double chi(double pU, double pL){
  if(pU - pL >= 0){
    return(1.0);
  }else{
    return(0.0);
  }
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

static void PrintFinalStats(void *kmem, int i)
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

  printf("\n\nFinal Statistics for Time Step %d\n",i);
  printf("nni    = %5ld    nli   = %5ld\n", nni, nli);
  printf("nfe    = %5ld    nfeSG = %5ld\n", nfe, nfeSG);
  printf("nps    = %5ld    npe   = %5ld     ncfl  = %5ld\n", nps, npe, ncfl);
}
