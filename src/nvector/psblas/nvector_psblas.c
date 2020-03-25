/* -----------------------------------------------------------------
 * Programmer(s): Fabio Durastante @ IAC-CNR
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
 * This is the implementation file for a parallel PSBLAS implementation
 * of the NVECTOR package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_mpi.h>

#include "psb_c_base.h"
#include "psb_base_cbind.h"

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 * Returns vector type ID. Used to identify vector implementation
 * from abstract N_Vector interface.
 */

N_Vector_ID N_VGetVectorID_PSBLAS(N_Vector v)
{
  return SUNDIALS_NVEC_CUSTOM;
}


/* ----------------------------------------------------------------
 * Function to create a new parallel vector with empty data array
 */

N_Vector N_VNewEmpty_PSBLAS(int ictxt, psb_c_descriptor *cdh)
{
  N_Vector v;
  N_Vector_Ops ops;
  N_VectorContent_PSBLAS content;
  sunindextype n, Nsum;

  /* Create vector */
  v = NULL;
  v = (N_Vector) malloc(sizeof *v);
  if (v == NULL) return(NULL);

  /* Create vector operation structure */
  ops = NULL;
  ops = (N_Vector_Ops) malloc(sizeof(struct _generic_N_Vector_Ops));
  if (ops == NULL) { free(v); return(NULL); }

  ops->nvgetvectorid     = N_VGetVectorID_PSBLAS;
  ops->nvclone           = N_VClone_PSBLAS;
  ops->nvcloneempty      = N_VCloneEmpty_PSBLAS;
  ops->nvdestroy         = N_VDestroy_PSBLAS;
  ops->nvspace           = N_VSpace_PSBLAS;
  ops->nvgetarraypointer = N_VGetArrayPointer_PSBLAS;
  ops->nvsetarraypointer = N_VSetArrayPointer_PSBLAS;

  /* standard vector operations */
  ops->nvlinearsum    = N_VLinearSum_PSBLAS;
  ops->nvconst        = N_VConst_PSBLAS;
  ops->nvprod         = N_VProd_PSBLAS;
  ops->nvdiv          = N_VDiv_PSBLAS;
  ops->nvscale        = N_VScale_PSBLAS;
  ops->nvabs          = N_VAbs_PSBLAS;
  ops->nvinv          = N_VInv_PSBLAS;
  ops->nvaddconst     = N_VAddConst_PSBLAS;
  ops->nvdotprod      = N_VDotProd_PSBLAS;
  ops->nvmaxnorm      = N_VMaxNorm_PSBLAS;
  ops->nvwrmsnormmask = N_VWrmsNormMask_PSBLAS;
  ops->nvwrmsnorm     = N_VWrmsNorm_PSBLAS;
  ops->nvmin          = N_VMin_PSBLAS;
  ops->nvwl2norm      = N_VWL2Norm_PSBLAS;
  ops->nvl1norm       = N_VL1Norm_PSBLAS;
  ops->nvcompare      = N_VCompare_PSBLAS;
  ops->nvinvtest      = N_VInvTest_PSBLAS;
  ops->nvconstrmask   = N_VConstrMask_PSBLAS;
  ops->nvminquotient  = N_VMinQuotient_PSBLAS;

  /* fused vector operations (optional, NULL means disabled by default) */
  ops->nvlinearcombination = NULL;
  ops->nvscaleaddmulti     = NULL;
  ops->nvdotprodmulti      = NULL;

  /* vector array operations (optional, NULL means disabled by default) */
  ops->nvlinearsumvectorarray         = NULL;
  ops->nvscalevectorarray             = NULL;
  ops->nvconstvectorarray             = NULL;
  ops->nvwrmsnormvectorarray          = NULL;
  ops->nvwrmsnormmaskvectorarray      = NULL;
  ops->nvscaleaddmultivectorarray     = NULL;
  ops->nvlinearcombinationvectorarray = NULL;

  /* Create content */
  content = NULL;
  content = (N_VectorContent_PSBLAS) malloc(sizeof(struct _N_VectorContent_PSBLAS));
  if (content == NULL) { free(ops); free(v); return(NULL); }

  /* Attach lengths and communicator */
  content->cdh           = cdh;
  content->ictxt         = ictxt;
  content->own_data      = SUNFALSE;
  content->pvec          = NULL;

  /* Attach content and ops */
  v->content = content;
  v->ops     = ops;

  return(v);
}

/* ----------------------------------------------------------------
 * Function to create a new parallel vector
 */

N_Vector N_VNew_PSBLAS(int ictxt, psb_c_descriptor *cdh)
{
  /*
  This function creates and allocates memory for a parallel vector
  on the PSBLAS context ictxt with the communicator cdh
  */
  N_Vector v;
  psb_c_dvector *pvec;

  v = NULL;
  v = N_VNewEmpty_PSBLAS(ictxt, cdh);
  if (v == NULL) return(NULL);

  /* Define new PSBLAS Vector */
  pvec = psb_c_new_dvector();
  /* Allocate mem space for the vector on the comunicator cdh */
  psb_c_dgeall(pvec,cdh);

  /* Attach data */
  NV_OWN_DATA_P(v) = SUNTRUE;
  NV_PVEC_P(v)     = pvec;

  return(v);
}

/* ----------------------------------------------------------------
 * Function to create a PSBLAS N_Vector with user data component:
 * This function does not allocate memory for pvec itsefl
 */

N_Vector N_VMake_PSBLAS(int ictxt, psb_c_descriptor *cdh, psb_i_t m, psb_l_t *irow,
                            double *val)
{
  N_Vector v;
  psb_i_t    local_length;

  v = NULL;
  v = N_VNewEmpty_PSBLAS(ictxt, cdh);
  if (v == NULL) return(NULL);

  psb_c_dgeins(m,irow,val,NV_PVEC_P(v),cdh);

  return(v);
}

/* ----------------------------------------------------------------
 * Function to assemble a PSBLAS N_Vector after the insertion of the
 * data is completed.
 */

void N_VAsb_PSBLAS(N_Vector v)
{
    psb_c_dgeasb(NV_PVEC_P(v),NV_DESCRIPTOR_P(v));
}

/* ----------------------------------------------------------------
 * Function to create an array of new parallel vectors:
 * this function creates (by cloning) an array of count parallel
 * vectors.
 */

N_Vector *N_VCloneVectorArray_PSBLAS(int count, N_Vector w)
{
  N_Vector *vs;
  int j;

  if (count <= 0) return(NULL);

  vs = NULL;
  vs = (N_Vector *) malloc(count * sizeof(N_Vector));
  if(vs == NULL) return(NULL);

  for (j = 0; j < count; j++) {
    vs[j] = NULL;
    vs[j] = N_VClone_PSBLAS(w);
    if (vs[j] == NULL) {
      N_VDestroyVectorArray_PSBLAS(vs, j-1);
      return(NULL);
    }
  }

  return(vs);
}

/* ----------------------------------------------------------------
 * Function to create an array of new parallel vectors with empty
 * (NULL) data array.
 */

N_Vector *N_VCloneVectorArrayEmpty_PSBLAS(int count, N_Vector w)
{
  N_Vector *vs;
  int j;

  if (count <= 0) return(NULL);

  vs = NULL;
  vs = (N_Vector *) malloc(count * sizeof(N_Vector));
  if(vs == NULL) return(NULL);

  for (j = 0; j < count; j++) {
    vs[j] = NULL;
    vs[j] = N_VCloneEmpty_PSBLAS(w);
    if (vs[j] == NULL) {
      N_VDestroyVectorArray_PSBLAS(vs, j-1);
      return(NULL);
    }
  }

  return(vs);
}

/* ----------------------------------------------------------------
 * Function to free an array created with N_VCloneVectorArray_PSBLAS
 */

void N_VDestroyVectorArray_PSBLAS(N_Vector *vs, int count)
{
  int j;

  for (j = 0; j < count; j++) N_VDestroy_PSBLAS(vs[j]);

  free(vs); vs = NULL;

  return;
}

/* ----------------------------------------------------------------
 * Function to return global vector length
 */

sunindextype N_VGetLength_PSBLAS(N_Vector v)
{
  return psb_c_cd_get_global_rows(NV_DESCRIPTOR_P(v));
}

/* ----------------------------------------------------------------
 * Function to return local vector length
 */

sunindextype N_VGetLocalLength_PSBLAS(N_Vector v)
{
  return psb_c_cd_get_local_rows(NV_DESCRIPTOR_P(v));
}

/* ----------------------------------------------------------------
 * Function to print the local data in a parallel vector to stdout
 */

void N_VPrint_PSBLAS(N_Vector x)
{
  N_VPrintFile_PSBLAS(x, stdout);
}

/* ----------------------------------------------------------------
 * Function to print the local data in a parallel vector to outfile
 */

void N_VPrintFile_PSBLAS(N_Vector x, FILE* outfile)
{
  sunindextype i, N;
  realtype *xd;

  xd = NULL;

  N  = N_VGetLocalLength_PSBLAS(x);
  xd = psb_c_dvect_get_cpy(NV_PVEC_P(x));

  for (i = 0; i < N; i++) {
#if defined(SUNDIALS_EXTENDED_PRECISION)
    fprintf(outfile, "%Lg\n", xd[i]);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    fprintf(outfile, "%g\n", xd[i]);
#else
    fprintf(outfile, "%g\n", xd[i]);
#endif
  }
  fprintf(outfile, "\n");

  return;
}

/*
 * -----------------------------------------------------------------
 * implementation of vector operations
 * -----------------------------------------------------------------
 */

N_Vector N_VCloneEmpty_PSBLAS(N_Vector w)
{
  N_Vector v;
  N_Vector_Ops ops;
  N_VectorContent_PSBLAS content;

  if (w == NULL) return(NULL);

  /* Create vector */
  v = NULL;
  v = (N_Vector) malloc(sizeof *v);
  if (v == NULL) return(NULL);

  /* Create vector operation structure */
  ops = NULL;
  ops = (N_Vector_Ops) malloc(sizeof(struct _generic_N_Vector_Ops));
  if (ops == NULL) { free(v); return(NULL); }

  ops->nvgetvectorid     = w->ops->nvgetvectorid;
  ops->nvclone           = w->ops->nvclone;
  ops->nvcloneempty      = w->ops->nvcloneempty;
  ops->nvdestroy         = w->ops->nvdestroy;
  ops->nvspace           = w->ops->nvspace;
  ops->nvgetarraypointer = w->ops->nvgetarraypointer;
  ops->nvsetarraypointer = w->ops->nvsetarraypointer;

  /* standard vector operations */
  ops->nvlinearsum    = w->ops->nvlinearsum;
  ops->nvconst        = w->ops->nvconst;
  ops->nvprod         = w->ops->nvprod;
  ops->nvdiv          = w->ops->nvdiv;
  ops->nvscale        = w->ops->nvscale;
  ops->nvabs          = w->ops->nvabs;
  ops->nvinv          = w->ops->nvinv;
  ops->nvaddconst     = w->ops->nvaddconst;
  ops->nvdotprod      = w->ops->nvdotprod;
  ops->nvmaxnorm      = w->ops->nvmaxnorm;
  ops->nvwrmsnormmask = w->ops->nvwrmsnormmask;
  ops->nvwrmsnorm     = w->ops->nvwrmsnorm;
  ops->nvmin          = w->ops->nvmin;
  ops->nvwl2norm      = w->ops->nvwl2norm;
  ops->nvl1norm       = w->ops->nvl1norm;
  ops->nvcompare      = w->ops->nvcompare;
  ops->nvinvtest      = w->ops->nvinvtest;
  ops->nvconstrmask   = w->ops->nvconstrmask;
  ops->nvminquotient  = w->ops->nvminquotient;

  /* fused vector operations */
  ops->nvlinearcombination = w->ops->nvlinearcombination;
  ops->nvscaleaddmulti     = w->ops->nvscaleaddmulti;
  ops->nvdotprodmulti      = w->ops->nvdotprodmulti;

  /* vector array operations */
  ops->nvlinearsumvectorarray         = w->ops->nvlinearsumvectorarray;
  ops->nvscalevectorarray             = w->ops->nvscalevectorarray;
  ops->nvconstvectorarray             = w->ops->nvconstvectorarray;
  ops->nvwrmsnormvectorarray          = w->ops->nvwrmsnormvectorarray;
  ops->nvwrmsnormmaskvectorarray      = w->ops->nvwrmsnormmaskvectorarray;
  ops->nvscaleaddmultivectorarray     = w->ops->nvscaleaddmultivectorarray;
  ops->nvlinearcombinationvectorarray = w->ops->nvlinearcombinationvectorarray;

  /* Create content */
  content = NULL;
  content = (N_VectorContent_PSBLAS) malloc(sizeof(struct _N_VectorContent_PSBLAS));
  if (content == NULL) { free(ops); free(v); return(NULL); }

  /* Attach lengths and communicator */
  content->cdh           = NV_DESCRIPTOR_P(w);
  content->ictxt         = NV_ICTXT_P(w);
  content->own_data      = SUNFALSE;
  content->pvec          = NULL;

  /* Attach content and ops */
  v->content = content;
  v->ops     = ops;

  return(v);
}

N_Vector N_VClone_PSBLAS(N_Vector w)
{
  N_Vector v;
  realtype *data;
  sunindextype local_length;

  v = NULL;
  v = N_VCloneEmpty_PSBLAS(w);
  if (v == NULL) return(NULL);

  NV_OWN_DATA_P(v) = SUNTRUE;
  NV_PVEC_P(v)     = NV_PVEC_P(w);

  return(v);
}

void N_VDestroy_PSBLAS(N_Vector v)
{
  if ((NV_OWN_DATA_P(v) == SUNTRUE) && (NV_PVEC_P(v) != NULL)) {
    free(NV_PVEC_P(v));
    NV_PVEC_P(v) = NULL;
  }
  free(v->content); v->content = NULL;
  free(v->ops); v->ops = NULL;
  free(v); v = NULL;

  return;
}

void N_VSpace_PSBLAS(N_Vector v, sunindextype *lrw, sunindextype *liw)
{
  int ictxt, npes, iam;

  ictxt = NV_ICTXT_P(v);
  psb_c_info(ictxt,&iam,&npes);

  *lrw = N_VGetLength_PSBLAS(v);
  *liw = 2*npes;

  return;
}

realtype *N_VGetArrayPointer_PSBLAS(N_Vector v)
{
  return((realtype *) NV_PVEC_P(v));
}

void N_VSetArrayPointer_PSBLAS(psb_c_dvector *v_data, N_Vector v)
{
  if (N_VGetLocalLength_PSBLAS(v) > 0) NV_PVEC_P(v) = v_data;

  return;
}

/* standard vector operations */

void N_VLinearSum_PSBLAS(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
{
/*Performs the operation z = ax + by, where a and b are realtype scalars and
x and y are of type N_Vector: z_i = a x_i + b y_i, i=0,...,n-1*/

  if(z == y){
    psb_c_dgeaxpby(a, NV_PVEC_P(x), b, NV_PVEC_P(z), NV_DESCRIPTOR_P(x));
  }else{
    psb_c_dgeaxpbyz(a, NV_PVEC_P(x), b, NV_PVEC_P(y), NV_PVEC_P(z), NV_DESCRIPTOR_P(x));
  }
  return;
}

void N_VConst_PSBLAS(realtype c, N_Vector z)
{
/*Sets all components of the N_Vector z to realtype c: z_i = c, i=0,...,n-1   */
  psb_i_t glob_row, ng;
  psb_l_t irow[1];
  double zt[1];
  zt[0]=c;
  ng = N_VGetLength_PSBLAS(z);
  for (glob_row=0; glob_row < ng; glob_row++) {
    irow[0]=glob_row;
    psb_c_dgeins(1,irow,zt,NV_PVEC_P(z) ,NV_DESCRIPTOR_P(z));
  }

  return;
}

void N_VProd_PSBLAS(N_Vector x, N_Vector y, N_Vector z){
/*Sets the N_Vector z to be the component-wise product of N_Vector inputs x and
y: z_i = x_i y_i, i=0,...,n-1 */
  psb_c_dgemlt2(1.0,NV_PVEC_P(x),NV_PVEC_P(y),0.0,NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

void N_VDiv_PSBLAS(N_Vector x, N_Vector y, N_Vector z){
/*Sets the N_Vector z to be the component-wise ratio of the N_vector inputs x
and y: z_i = x_i/y_i, i=0,...,n-1. The y_i are NOT tested for 0 values. It should
ONLY be called with an y that is guaranted to have all nonzero components*/
  psb_c_dgediv2(NV_PVEC_P(x),NV_PVEC_P(y),NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

void N_VScale_PSBLAS(realtype c, N_Vector x, N_Vector z){
/* Scales the N_Vector x by the realtype scalar c and returns the results in z.
*/
  psb_c_dgescal(NV_PVEC_P(x),c,NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

void N_VAbs_PSBLAS(N_Vector x, N_Vector z){
/*Sets the component of the N_Vector z to be the absolute values of the compoents
of the N_Vector x: y_i = |x_i|, i=0,...,n-1*/
  psb_c_dgeabs(NV_PVEC_P(x),NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

void N_VInv_PSBLAS(N_Vector x, N_Vector z){
/*Sets the N_Vector z to be the component-wise inverse of the N_vector inputs x
and y: z_i = 1/y_i, i=0,...,n-1. The y_i are NOT tested for 0 values. It should
ONLY be called with an y that is guaranted to have all nonzero components*/
  psb_c_dgeinv(NV_PVEC_P(x),NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

void N_VAddConst_PSBLAS(N_Vector x, realtype b, N_Vector z){
/*Adds the realtype scalar b to all components of x and returns the result in
the N_Vector z: z_i = x_i + b_i*/
  psb_c_dgeaddconst(NV_PVEC_P(x),b,NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  return;
}

realtype N_VDotProd_PSBLAS(N_Vector x, N_Vector y){
/* Returns the value of the ordinary dot product of x and y                   */
  return(psb_c_dgedot(NV_PVEC_P(x),NV_PVEC_P(y),NV_DESCRIPTOR_P(x)));
}

realtype N_VMaxNorm_PSBLAS(N_Vector x){
/* Returns the maximum norm of the N_Vector x*/
  return(psb_c_dgeamax(NV_PVEC_P(x),NV_DESCRIPTOR_P(x)));
}

realtype N_VWrmsNorm_PSBLAS(N_Vector x, N_Vector w){
  return(psb_c_dgenrm2_weight(NV_PVEC_P(x),NV_PVEC_P(w),NV_DESCRIPTOR_P(x)));
}

realtype N_VWrmsNormMask_PSBLAS(N_Vector x, N_Vector w, N_Vector id){
  return(psb_c_dgenrm2_weightmask(NV_PVEC_P(x),NV_PVEC_P(w),NV_PVEC_P(id),NV_DESCRIPTOR_P(x)));
}

realtype N_VMin_PSBLAS(N_Vector x){
/* Returns the minimum of vector x*/
  return(psb_c_dgemin(NV_PVEC_P(x),NV_DESCRIPTOR_P(x)));
}

realtype N_VWL2Norm_PSBLAS(N_Vector x, N_Vector w){
/* Returns the weighted 2-norm of the vector x with respect to the weights w,
i.e., res = ||w.*x||_2 = sqrt( sum_{j} (w_j x_j)^2 )*/
  return(psb_c_dgenrm2_weight(NV_PVEC_P(x),NV_PVEC_P(w),NV_DESCRIPTOR_P(x)));
}

realtype N_VL1Norm_PSBLAS(N_Vector x){
/* Returns the 1-norm of the vector x */
  return(psb_c_dspnrmi(NV_PVEC_P(x),NV_DESCRIPTOR_P(x)));
}

void N_VCompare_PSBLAS(realtype c, N_Vector x, N_Vector z){
/* Compare the vector x with the real constant c and put the result in z, i.e.,
z_i = 1 if |x_i| < c else z_i = 0 for i=1,...,length(x)*/
   psb_c_dgecmp(NV_PVEC_P(x),c,NV_PVEC_P(z),NV_DESCRIPTOR_P(x));

   return;
}

booleantype N_VInvTest_PSBLAS(N_Vector x, N_Vector z){
/* Computes z_i = 1/x_i iff x_i != 0, returns true if every test was positive,
otherwise returns false */
  bool ret;
  psb_i_t info;

  info = psb_c_dgeinv_check(NV_PVEC_P(x),NV_PVEC_P(z),NV_DESCRIPTOR_P(x));
  ret = info;

  return(ret);
}

booleantype N_VConstrMask_PSBLAS(N_Vector c, N_Vector x, N_Vector m){
  bool t;
  psb_c_dmask(NV_PVEC_P(c),NV_PVEC_P(x),NV_PVEC_P(m), t, NV_DESCRIPTOR_P(x));

  return(t);
}

realtype N_VMinQuotient_PSBLAS(N_Vector num, N_Vector denom){
/* Returns min_i num_i/denom_i for the denom_i != 0 */
  return(psb_c_dminquotient(NV_PVEC_P(num),NV_PVEC_P(denom),NV_DESCRIPTOR_P(num)));
}

/*
* -----------------------------------------------------------------
* fused vector operations
* -----------------------------------------------------------------
*/
int N_VLinearCombination_PSBLAS(int nvec, realtype* c, N_Vector* V,N_Vector z){
  psb_i_t      ierr;
  sunindextype j, N;

  /* invalid number of vectors */
  if (nvec < 1) return(-1);

  /* should have called N_VScale */
  if (nvec == 1) {
    N_VScale_PSBLAS(c[0], V[0], z);
    return(0);
  }

  /* should have called N_VLinearSum */
  if (nvec == 2) {
    N_VLinearSum_PSBLAS(c[0], V[0], c[1], V[1], z);
    return(0);
  }

  for(j=0;j<nvec;j++){
    ierr = psb_c_dgeaxpby( c[j], NV_PVEC_P(V[j]), 1.0, NV_PVEC_P(z), NV_DESCRIPTOR_P(z));
  }

  return(ierr);

};
int N_VScaleAddMulti_PSBLAS(int nvec, realtype* a, N_Vector x,N_Vector* Y, N_Vector* Z){

  int j;

  /* invalid number of vectors */
  if (nvec < 1) return(-1);

  /* should have called N_VLinearSum */
  if (nvec == 1) {
    N_VLinearSum_PSBLAS(a[0], x, 1.0, Y[0], Z[0]);
    return(0);
  }

  for(j=0;j<nvec;j++){
    N_VLinearSum_PSBLAS(a[j], x, 1.0, Y[j], Z[j]);
    return(0);
  }


  return(1);
};
int N_VDotProdMulti_PSBLAS(int nvec, N_Vector x, N_Vector *Y, realtype* dotprods){

  int j;
  for(j = 0; j < nvec; j++){
    dotprods[j] = N_VDotProd_PSBLAS(x, Y[j]);
  }

 return(0);
};

/*
* -----------------------------------------------------------------
* vector array operations
* -----------------------------------------------------------------
*/
int N_VLinearSumVectorArray_PSBLAS(int nvec,realtype a, N_Vector* X, realtype b, N_Vector* Y, N_Vector* Z){

 int j;

 for(j=0;j<nvec;j++){
   psb_c_dgeaxpbyz(a,NV_PVEC_P(X[j]),b,NV_PVEC_P(Y[j]),NV_PVEC_P(Z[j]),NV_DESCRIPTOR_P(X[j]));
 }

 return(0);

};
int N_VScaleVectorArray_PSBLAS(int nvec, realtype* c, N_Vector* X, N_Vector* Z){

  int j;

  for(j=0;j<nvec;j++){
    psb_c_dgescal(NV_PVEC_P(X[j]),c[j],NV_PVEC_P(Z[j]),NV_DESCRIPTOR_P(X[j]));
  }

  return(0);
};

int N_VConstVectorArray_PSBLAS(int nvecs, realtype c, N_Vector* Z){
  int j;

  for(j=0;j<nvecs;j++){
    N_VConst_PSBLAS(c, Z[j]);
  }

  return(0);
};

int N_VWrmsNormVectorArray_PSBLAS(int nvecs, N_Vector* X, N_Vector* W, realtype* nrm){
  int j;
  for(j=0; j<nvecs; j++){
    nrm[j] = N_VWrmsNorm(X[j],W[j]);
  }
  return(1);
};
int N_VWrmsNormMaskVectorArray_PSBLAS(int nvec, N_Vector* X, N_Vector* W, N_Vector id,realtype* nrm){
  int j;

  for(j=0; j<nvec; j++){
    nrm[j] = N_VWrmsNormMask_PSBLAS(X[j],W[j],id);
  }
  return(1);
};
int N_VScaleAddMultiVectorArray_PSBLAS(int nvec, int nsum, realtype* a, N_Vector* X, N_Vector** Y, N_Vector** Z){
  int          i, j, retval;
  sunindextype k, N;

  N_Vector*    YY;
  N_Vector*    ZZ;

  /* ---------------------------
   * Special cases for nvec == 1
   * --------------------------- */

  if (nvec == 1) {

    /* should have called N_VLinearSum */
    if (nsum == 1) {
      N_VLinearSum_PSBLAS(a[0], X[0], 1.0, Y[0][0], Z[0][0]);
      return(0);
    }

    /* should have called N_VScaleAddMulti */
    YY = (N_Vector *) malloc(nsum * sizeof(N_Vector));
    ZZ = (N_Vector *) malloc(nsum * sizeof(N_Vector));

    for (j=0; j<nsum; j++) {
      YY[j] = Y[j][0];
      ZZ[j] = Z[j][0];
    }

    retval = N_VScaleAddMulti_PSBLAS(nsum, a, X[0], YY, ZZ);

    free(YY);
    free(ZZ);
    return(retval);
  }

  /* --------------------------
   * Special cases for nvec > 1
   * -------------------------- */

  /* should have called N_VLinearSumVectorArray */
  if (nsum == 1) {
    retval = N_VLinearSumVectorArray_PSBLAS(nvec, a[0], X, 1.0, Y[0], Z[0]);
    return(retval);
  }

    /*
    * Y[i][j] += a[i] * x[j]
    */

    if (Y == Z) {

      for (i=0; i<nvec; i++) {
        for (j=0; j<nsum; j++){
          psb_c_dgeaxpby(a[i],NV_PVEC_P(X[j]),1.0,NV_PVEC_P(Y[i][j]),NV_DESCRIPTOR_P(X[j]));
        }
      }

      return(0);
    }
    /*
    * Z[i][j] = Y[i][j] + a[i] * x[j]
    */
    for (i=0; i<nvec; i++) {
      for (j=0; j<nsum; j++){
        psb_c_dgeaxpbyz(a[i],NV_PVEC_P(X[j]),1.0,NV_PVEC_P(Y[i][j]),NV_PVEC_P(Z[i][j]),NV_DESCRIPTOR_P(X[j]));
      }
    }
    return(0);
};
int N_VLinearCombinationVectorArray_PSBLAS(int nvec, int nsum, realtype* c, N_Vector** X, N_Vector* Z){
 return(1);
};

/*
 * -----------------------------------------------------------------
 * Enable / Disable fused and vector array operations
 * -----------------------------------------------------------------
 */

int N_VEnableFusedOps_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  if (tf) {
    /* enable all fused vector operations */
    v->ops->nvlinearcombination = N_VLinearCombination_PSBLAS;
    v->ops->nvscaleaddmulti     = N_VScaleAddMulti_PSBLAS;
    v->ops->nvdotprodmulti      = N_VDotProdMulti_PSBLAS;
    /* enable all vector array operations */
    v->ops->nvlinearsumvectorarray         = N_VLinearSumVectorArray_PSBLAS;
    v->ops->nvscalevectorarray             = N_VScaleVectorArray_PSBLAS;
    v->ops->nvconstvectorarray             = N_VConstVectorArray_PSBLAS;
    v->ops->nvwrmsnormvectorarray          = N_VWrmsNormVectorArray_PSBLAS;
    v->ops->nvwrmsnormmaskvectorarray      = N_VWrmsNormMaskVectorArray_PSBLAS;
    v->ops->nvscaleaddmultivectorarray     = N_VScaleAddMultiVectorArray_PSBLAS;
    v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_PSBLAS;
  } else {
    /* disable all fused vector operations */
    v->ops->nvlinearcombination = NULL;
    v->ops->nvscaleaddmulti     = NULL;
    v->ops->nvdotprodmulti      = NULL;
    /* disable all vector array operations */
    v->ops->nvlinearsumvectorarray         = NULL;
    v->ops->nvscalevectorarray             = NULL;
    v->ops->nvconstvectorarray             = NULL;
    v->ops->nvwrmsnormvectorarray          = NULL;
    v->ops->nvwrmsnormmaskvectorarray      = NULL;
    v->ops->nvscaleaddmultivectorarray     = NULL;
    v->ops->nvlinearcombinationvectorarray = NULL;
  }

  /* return success */
  return(0);
}


int N_VEnableLinearCombination_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvlinearcombination = N_VLinearCombination_PSBLAS;
  else
    v->ops->nvlinearcombination = NULL;

  /* return success */
  return(0);
}

int N_VEnableScaleAddMulti_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvscaleaddmulti = N_VScaleAddMulti_PSBLAS;
  else
    v->ops->nvscaleaddmulti = NULL;

  /* return success */
  return(0);
}

int N_VEnableDotProdMulti_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvdotprodmulti = N_VDotProdMulti_PSBLAS;
  else
    v->ops->nvdotprodmulti = NULL;

  /* return success */
  return(0);
}

int N_VEnableLinearSumVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvlinearsumvectorarray = N_VLinearSumVectorArray_PSBLAS;
  else
    v->ops->nvlinearsumvectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableScaleVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvscalevectorarray = N_VScaleVectorArray_PSBLAS;
  else
    v->ops->nvscalevectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableConstVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvconstvectorarray = N_VConstVectorArray_PSBLAS;
  else
    v->ops->nvconstvectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableWrmsNormVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvwrmsnormvectorarray = N_VWrmsNormVectorArray_PSBLAS;
  else
    v->ops->nvwrmsnormvectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableWrmsNormMaskVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvwrmsnormmaskvectorarray = N_VWrmsNormMaskVectorArray_PSBLAS;
  else
    v->ops->nvwrmsnormmaskvectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableScaleAddMultiVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_PSBLAS;
  else
    v->ops->nvscaleaddmultivectorarray = NULL;

  /* return success */
  return(0);
}

int N_VEnableLinearCombinationVectorArray_PSBLAS(N_Vector v, booleantype tf)
{
  /* check that vector is non-NULL */
  if (v == NULL) return(-1);

  /* check that ops structure is non-NULL */
  if (v->ops == NULL) return(-1);

  /* enable/disable operation */
  if (tf)
    v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_PSBLAS;
  else
    v->ops->nvlinearcombinationvectorarray = NULL;

  /* return success */
  return(0);
}
