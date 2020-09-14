
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

 int main(int argc, char *argv[]){

    return(0);
}
