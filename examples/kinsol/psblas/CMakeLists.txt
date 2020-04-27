# ---------------------------------------------------------------
# Programmer:  F. Durastante @ IAC-CNR
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------
# CMakeLists.txt file for KINSOL psblas examples
# ---------------------------------------------------------------

# Example lists are tuples "name\;nodes\;tasks\;type" where the
# type is develop for examples excluded from 'make test' in releases

# Examples using SUNDIALS linear solvers
SET(KINSOL_examples
  "kinFoodWeb_kry_psblas\;1\;4\;develop"
  )

# Dependencies for kinsol examples
SET(kinsol_psblas_dependencies
  sundials_nvector
  sundials_matrix
  sundials_linearsolver
)

# Add source directory to include directories
include_directories(. ..)

# Add PSBLAS flag
INCLUDE(${sundials_SOURCE_DIR}/src/sunlinsol/psblas/makeincinputcmake)

INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

# Specify libraries to link against (through the target that was used to
# generate them) based on the value of the variable LINK_LIBRARY_TYPE
IF(LINK_LIBRARY_TYPE MATCHES "static")
  SET(KINSOL_LIB sundials_kinsol_static)
  SET(NVECS_LIB sundials_nvecpsblas_static)
  SET(SUNMATS_LIB sundials_sunmatrixpsblas_static)
  SET(SUNLINSOL_LIB sundials_sunlinsolpsblas_static)
ELSE()
  SET(KINSOL_LIB sundials_kinsol_shared)
  SET(NVECS_LIB sundials_nvecpsblas_shared)
  SET(SUNMATS_LIB sundials_sunmatrixpsblas_shared)
  SET(SUNLINSOL_LIB sundials_sunlinsolpsblas_shared)
ENDIF()

# Set-up linker flags and link libraries
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY ${PSBCDEFINES})

# Set-up linker flags and link libraries
SET(SUNDIALS_LIBS ${KINSOL_LIB} ${NVECS_LIB} ${EXTRA_LINK_LIBS} ${SUNMATS_LIB} ${SUNLINSOL_LIB})

# PSBLAS LINK DIRECTORIES
LINK_DIRECTORIES(${PSBLAS_LIBRARY_DIR})
LINK_LIBRARIES(${PSBLDLIBS} ${PSBLAS_LIBS} -lgfortran)


IF(MPI_C_COMPILER)
  # use MPI wrapper as the compiler
  SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
ELSE()
  # add MPI_INCLUDE_PATH to include directories
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
ENDIF()

# Add the build and install targets for each example
FOREACH(example_tuple ${KINSOL_examples})

  # parse the example tuple
  LIST(GET example_tuple 0 example)
  LIST(GET example_tuple 1 number_of_nodes)
  LIST(GET example_tuple 2 number_of_tasks)
  LIST(GET example_tuple 3 example_type)

  # example source files
  ADD_EXECUTABLE(${example} ${example}.c
    ../../../../src/sundials/sundials_linearsolver.c
    ../../../../src/sundials/sundials_nvector.c
    ../../../../src/sundials/sundials_matrix.c)

  SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

  # add example to regression tests
  SUNDIALS_ADD_TEST(${example} ${example}
    MPI_NPROCS ${number_of_tasks}
    ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    ANSWER_FILE ${example}.out
    EXAMPLE_TYPE ${example_type})

  # libraries to link against
  TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS})

  IF(NOT MPI_C_COMPILER)
    TARGET_LINK_LIBRARIES(${example} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARIES})
  ENDIF()

  # install example source and out files
  IF(EXAMPLES_INSTALL)
    INSTALL(FILES ${example}.c ${example}.out
      DESTINATION ${EXAMPLES_INSTALL_PATH}/kinsol/psblas)
  ENDIF()

ENDFOREACH(example_tuple ${KINSOL_examples})


# create Makfile and CMakeLists.txt for examples
IF(EXAMPLES_INSTALL)

  # Install the README file
  INSTALL(FILES README
    ../../../src/sundials/sundials_linearsolver.c
    ../../../src/sundials/sundials_matrix.c
    ../../../src/sundials/sundials_nvector.c
    DESTINATION ${EXAMPLES_INSTALL_PATH}/kinsol/psblas)

  # Prepare substitution variables for Makefile and/or CMakeLists templates
  SET(SOLVER "KINSOL")
  SET(SOLVER_LIB "sundials_kinsol")

  EXAMPLES2STRING(KINSOL_examples EXAMPLES)
  EXAMPLES2STRING(kinsol_psblas_dependencies EXAMPLES_DEPENDENCIES)

  # Regardless of the platform we're on, we will generate and install
  # CMakeLists.txt file for building the examples. This file  can then
  # be used as a template for the user's own programs.

  # generate CMakelists.txt in the binary directory
  CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_psblas_C_ex.in
    ${PROJECT_BINARY_DIR}/examples/kinsol/psblas/CMakeLists.txt
    @ONLY
    )

  # install CMakelists.txt
  INSTALL(
    FILES ${PROJECT_BINARY_DIR}/examples/kinsol/psblas/CMakeLists.txt
    DESTINATION ${EXAMPLES_INSTALL_PATH}/kinsol/psblas
    )

  # On UNIX-type platforms, we also  generate and install a makefile for
  # building the examples. This makefile can then be used as a template
  # for the user's own programs.

  IF(UNIX)
    # generate Makefile and place it in the binary dir
    CONFIGURE_FILE(
      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_psblas_C_ex.in
      ${PROJECT_BINARY_DIR}/examples/kinsol/psblas/Makefile_ex
      @ONLY
      )
    # install the configured Makefile_ex as Makefile
    INSTALL(
      FILES ${PROJECT_BINARY_DIR}/examples/kinsol/psblas/Makefile_ex
      DESTINATION ${EXAMPLES_INSTALL_PATH}/kinsol/psblas
      RENAME Makefile
      )
  ENDIF(UNIX)

ENDIF(EXAMPLES_INSTALL)
