# ---------------------------------------------------------------
# Programmer(s): F. Durastante @ SMU
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
# CMakeLists.txt file for sunlinsol PSBLAS parallel examples
# ---------------------------------------------------------------

# Example lists are tuples "name\;args\;nodes\;tasks\;type" where the
# type is develop for examples excluded from 'make test' in releases

# Examples using the SUNDIALS SPBCGS linear solver
SET(sunlinsol_psblas_examples
  "test_sunlinsol_psblas\;100 1 50 1e-3 0\;1\;4\;"
  "test_sunlinsol_psblas\;100 2 50 1e-3 0\;1\;4\;"
  )

# Dependencies for nvector examples
SET(sunlinsol_psblas_dependencies
  test_sunlinsol
  sundials_nvector
  sundials_matrix
  sundials_linearsolver
  )

# Add source directory to include directories
INCLUDE_DIRECTORIES(. ..)

# Add PSBLAS flag
INCLUDE(${sundials_SOURCE_DIR}/src/sunlinsol/psblas/makeincinputcmake)

# Specify libraries to link against (through the target that was used to
# generate them) based on the value of the variable LINK_LIBRARY_TYPE
IF(LINK_LIBRARY_TYPE MATCHES "static")
  SET(NVECS_LIB sundials_nvecpsblas_static)
  SET(SUNMATS_LIB sundials_sunmatrixpsblas_static)
  SET(SUNLINSOL_LIB sundials_sunlinsolpsblas_static)
ELSE()
  SET(NVECS_LIB sundials_nvecpsblas_shared)
  SET(SUNMATS_LIB sundials_sunmatrixpsblas_shared)
  SET(SUNLINSOL_LIB sundials_sunlinsolpsblas_shared)
ENDIF()

# Set-up linker flags and link libraries
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY ${PSBCDEFINES})

# Set-up linker flags and link libraries
SET(SUNDIALS_LIBS ${NVECS_LIB} ${EXTRA_LINK_LIBS} ${SUNMATS_LIB} ${SUNLINSOL_LIB})

# PSBLAS LINK DIRECTORIES
LINK_DIRECTORIES(${PSBLAS_LIBRARY_DIR})
LINK_LIBRARIES(${PSBLDLIBS} ${PSBLAS_LIBS} -lgfortran -lmld_cbind -lmld_prec)


IF(MPI_C_COMPILER)
  # use MPI wrapper as the compiler
  SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
ELSE()
  # add MPI_INCLUDE_PATH to include directories
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
ENDIF()


# Add the build and install targets for each example
FOREACH(example_tuple ${sunlinsol_psblas_examples})

  # parse the example tuple
  LIST(GET example_tuple 0 example)
  LIST(GET example_tuple 1 example_args)
  LIST(GET example_tuple 2 number_of_nodes)
  LIST(GET example_tuple 3 number_of_tasks)
  LIST(GET example_tuple 4 example_type)

  # This is used to get around DLL linkage issue since we are
  # manually including sundials_nvector.c here, which is normally in
  # a library that is included.  If this is not set build system
  # thinks nvector is externally linked.
  IF(WIN32)
    ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY)
  ENDIF(WIN32)

  # check if this example has already been added, only need to add
  # example source files once for testing with different inputs
  IF(NOT TARGET ${example})
    # example source files
    ADD_EXECUTABLE(${example} ${example}.c
      ../test_sunlinsol.c
      ../../../../src/sundials/sundials_linearsolver.c
      ../../../../src/sundials/sundials_nvector.c)

    # folder to organize targets in an IDE
    SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

    # libraries to link against
    TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS})

    IF(NOT MPI_C_COMPILER)
      TARGET_INCLUDE_DIRECTORIES(${example} ${MPI_INCLUDE_PATH})
      TARGET_LINK_LIBRARIES(${example} ${MPI_LIBRARIES})
    ENDIF()
  ENDIF()

  # check if example args are provided and set the test name
  IF("${example_args}" STREQUAL "")
    SET(test_name ${example})
  ELSE()
    STRING(REGEX REPLACE " " "_" test_name ${example}_${example_args})
  ENDIF()

  # add example to regression tests
  SUNDIALS_ADD_TEST(${test_name} ${example}
    TEST_ARGS ${example_args}
    MPI_NPROCS ${number_of_tasks}
    EXAMPLE_TYPE ${example_type}
    NODIFF)

  # install example source files
  IF(EXAMPLES_INSTALL)
    INSTALL(FILES ${example}.c
      mlde.inp
      ../test_sunlinsol.h
      ../test_sunlinsol.c
      ../../../src/sundials/sundials_linearsolver.c
      ../../../src/sundials/sundials_nvector.c
      ../../../src/sundials/sundials_matrix.c
      DESTINATION ${EXAMPLES_INSTALL_PATH}/sunlinsol/psblas)
  ENDIF()

ENDFOREACH(example_tuple ${sunlinsol_psblas_examples})


IF(EXAMPLES_INSTALL)

  # Install the README file
  INSTALL(FILES DESTINATION ${EXAMPLES_INSTALL_PATH}/sunlinsol/psblas)

  # Prepare substitution variables for Makefile and/or CMakeLists templates
  SET(SOLVER_LIB "sundials_sunlinsolpsblas")

  EXAMPLES2STRING(sunlinsol_psblas_examples EXAMPLES)
  EXAMPLES2STRING(sunlinsol_psblas_dependencies EXAMPLES_DEPENDENCIES)

  # Regardless of the platform we're on, we will generate and install
  # CMakeLists.txt file for building the examples. This file  can then
  # be used as a template for the user's own programs.

  # generate CMakelists.txt in the binary directory
  CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_psblas_C_ex.in
    ${PROJECT_BINARY_DIR}/examples/sunlinsol/psblas/CMakeLists.txt
    @ONLY
    )

  # install CMakelists.txt
  INSTALL(
    FILES ${PROJECT_BINARY_DIR}/examples/sunlinsol/psblas/CMakeLists.txt
    DESTINATION ${EXAMPLES_INSTALL_PATH}/sunlinsol/psblas
    )

  # On UNIX-type platforms, we also  generate and install a makefile for
  # building the examples. This makefile can then be used as a template
  # for the user's own programs.

  IF(UNIX)
    # generate Makefile and place it in the binary dir
    CONFIGURE_FILE(
      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_psblas_C_ex.in
      ${PROJECT_BINARY_DIR}/examples/sunlinsol/psblas/Makefile_ex
      @ONLY
      )
    # install the configured Makefile_ex as Makefile
    INSTALL(
      FILES ${PROJECT_BINARY_DIR}/examples/sunlinsol/psblas/Makefile_ex
      DESTINATION ${EXAMPLES_INSTALL_PATH}/sunlinsol/psblas
      RENAME Makefile
      )
  ENDIF(UNIX)

ENDIF(EXAMPLES_INSTALL)
