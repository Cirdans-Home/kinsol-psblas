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
# CMakeLists.txt file for the psblas SUNMatrix library

INSTALL(CODE "MESSAGE(\"\nInstall SUNMATRIX_PSBLAS\n\")")

# This part will be removed when PSBLAS/MLD2P4 and the other libraries will be
# installed by using CMake, this works as a patch for including the information
# contained in the Make.inc files (Compile flags etc.), I know it is very
# horrible indeed
EXECUTE_PROCESS(COMMAND ./make2cmakeset.sh ${PSBLAS_INCLUDE_DIR} ${sundials_SOURCE_DIR}/src/nvector/psblas WORKING_DIRECTORY ${sundials_SOURCE_DIR}/src/nvector/psblas OUTPUT_QUIET ERROR_QUIET)
INCLUDE(${sundials_SOURCE_DIR}/src/nvector/psblas/makeincinputcmake)

IF(MPI_C_COMPILER)
  # use MPI wrapper as the compiler
  SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
ELSE()
  # add MPI_INCLUDE_PATH to include directories
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
ENDIF()

# Add variable sunmatrixpsblas_SOURCES with the sources for the SUNMATRIXPSBLAS lib
SET(sunmatrixpsblas_SOURCES sunmatrix_psblas.c)

# Add variable shared_SOURCES with the common SUNDIALS sources which will
# also be included in the SUNMATRIXPSBLAS library
SET(shared_SOURCES
  ${sundials_SOURCE_DIR}/src/sundials/sundials_matrix.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_math.c
  )

# Add variable sunmatrixpsblas_HEADERS with the exported SUNMATRIXPSBLAS header files
SET(sunmatrixpsblas_HEADERS
  ${sundials_SOURCE_DIR}/include/sunmatrix/sunmatrix_psblas.h
  )

# Add source directory to include directories
INCLUDE_DIRECTORIES(.)

# Define C preprocessor flag -DBUILD_SUNDIALS_LIBRARY
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY ${PSBCDEFINES})

# PSBLAS LINK DIRECTORIES
LINK_DIRECTORIES(${PSBLAS_LIBRARY_DIR})
LINK_LIBRARIES(${PSBLDLIBS} ${PSBLAS_LIBS})

# Rules for building and installing the static library:
#  - Add the build target for the SUNMATRIXPSBLAS library
#  - Set the library name and make sure it is not deleted
#  - Install the SUNMATRIXPSBLAS library
IF(BUILD_STATIC_LIBS)
  ADD_LIBRARY(sundials_sunmatrixpsblas_static STATIC ${sunmatrixpsblas_SOURCES} ${shared_SOURCES})
  SET_TARGET_PROPERTIES(sundials_sunmatrixpsblas_static
    PROPERTIES OUTPUT_NAME sundials_sunmatrixpsblas CLEAN_DIRECT_OUTPUT 1)
  INSTALL(TARGETS sundials_sunmatrixpsblas_static DESTINATION ${CMAKE_INSTALL_LIBDIR})
ENDIF(BUILD_STATIC_LIBS)

# Rules for building and installing the shared library:
#  - Add the build target for the SUNMATRIXPSBLAS library
#  - Set the library name and make sure it is not deleted
#  - Set VERSION and SOVERSION for shared libraries
#  - Install the SUNMATRIXPSBLAS library
IF(BUILD_SHARED_LIBS)
  ADD_LIBRARY(sundials_sunmatrixpsblas_shared SHARED ${sunmatrixpsblas_SOURCES} ${shared_SOURCES})

  IF(UNIX)
    TARGET_LINK_LIBRARIES(sundials_sunmatrixpsblas_shared m)
  ENDIF()

  SET_TARGET_PROPERTIES(sundials_sunmatrixpsblas_shared
    PROPERTIES OUTPUT_NAME sundials_sunmatrixpsblas CLEAN_DIRECT_OUTPUT 1)
  SET_TARGET_PROPERTIES(sundials_sunmatrixpsblas_shared
    PROPERTIES VERSION ${sunmatrixlib_VERSION} SOVERSION ${sunmatrixlib_SOVERSION})
  INSTALL(TARGETS sundials_sunmatrixpsblas_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
ENDIF(BUILD_SHARED_LIBS)

# Install the SUNMATRIXPSBLAS header files
INSTALL(FILES ${sunmatrixpsblas_HEADERS} DESTINATION include/sunmatrix)

#
MESSAGE(STATUS "Added SUNMATRIX_PSBLAS module")
