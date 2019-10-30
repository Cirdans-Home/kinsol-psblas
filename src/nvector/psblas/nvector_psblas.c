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
 * This is the implementation file for a parallel MPI implementation
 * of the NVECTOR package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_mpi.h>

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
  on the PSBLAS context ictxt with the comunicator cdh
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
 * This function does not allocate memory for v_data itsefl
 */

N_Vector N_VMake_PSBLAS(int ictxt, psb_c_descriptor *cdh,
                          psb_c_dvector *v_data)
{
  N_Vector v;
  psb_i_t    local_length; 

  v = NULL;
  v = N_VNewEmpty_PSBLAS(ictxt, cdh);
  if (v == NULL) return(NULL);

  /* Get local length of the vector */ 
  local_length = psb_c_cd_get_local_rows(cdh);

  if (local_length > 0) {
    /* Attach data */
    NV_OWN_DATA_P(v) = SUNFALSE;
    NV_PVEC_P(v)     = v_data;
  }

  return(v);
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

void N_VLinearSum_PSBLAS(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
{

  z = N_VClone_PSBLAS(y);
  psb_c_dgeaxpby(a, NV_PVEC_P(x), b, NV_PVEC_P(z), NV_DESCRIPTOR_P(x));  

  return;
}
