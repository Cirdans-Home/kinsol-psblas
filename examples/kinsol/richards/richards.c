
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

 /* ------------------------------------------------
 Auxiliary functions for KINSOL
 -------------------------------------------------*/
 static int funcprpr(N_Vector u, N_Vector fval, void *user_data);
 static int jac(N_Vector y, N_Vector f, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2);

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

 #define NBMAX       20

 int main(int argc, char *argv[]){

    /* BLOCK data distribution */
    psb_l_t ng,nl,*vl;
    psb_i_t nb;
    /* Input from file */
    psb_i_t nparms,idim;
    /* Parallel Environment */
    psb_i_t ictxt,iam,np,info;
    psb_c_descriptor *cdh;

    /* Auxiliary variable */
    int i,k;


    ictxt = psb_c_init();
    psb_c_info(ictxt,&iam,&np);
    fprintf(stdout,"Initialization of the Richards miniapp I'm %d of %d\n",iam,np);
    fflush(stdout);

    /* Read Problem Settings from file */
    if (iam == 0) {
      get_iparm(stdin,&nparms);
      get_iparm(stdin,&idim);
   }
   psb_c_ibcast(ictxt,1,&nparms,0);
   psb_c_ibcast(ictxt,1,&idim,0);

    /* Perform a 3D BLOCK data distribution */
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


    psb_c_barrier(ictxt);
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
/* This function returns the evaluation fval = \Phi(u;parameters) to march the
   Newton method.                                                             */


  return(0);
}

static int jac(N_Vector yvec, N_Vector fvec, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2)
{
  /* This function returns the evaluation of the Jacobian of the system upon
  request of the Newton method.                                               */

  return(0);
}
