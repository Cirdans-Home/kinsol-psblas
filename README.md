# KINSOL-PSBLAS
KINSOL library with PSBLAS linear solvers

KINSOL is a general-purpose nonlinear system solver based on Newton-Krylov solver technology, together with a fixed point iteration method (kinsol version >= v.2.8.0).

The present version interfaces KINSOL with the functionalities for handling distributed (sparse) linear algebra of the [PSBLAS](https://github.com/sfilippone/psblas3) and [MLD2P4](https://github.com/sfilippone/mld2p4-2) libraries. Specifically, it furnishes new versions of the
* N_Vector
* SUNMatrix
* SUNLinsol
modules that use the functionalities offered by the PSBLAS and MLD2P4 libraries. 

The main objective of this code is enabling the KINSOL Newton-Krylov solver for the usage of the Krylov solvers and the multilevel preconditioners included in the PSBLAS/MLD2P4 library.

## Funding
Work supported by the EC under the Horizon 2020 Project “Energy oriented Centre of Excellence for computing applications” (EoCoE II), Project ID: 824158
