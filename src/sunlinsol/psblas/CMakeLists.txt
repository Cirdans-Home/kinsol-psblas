# ---------------------------------------------------------------
# Programmer:  F. Durastante @ IAC CNR
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
# CMakeLists.txt file for the PSBLASFCG SUNLinearSolver library

INSTALL(CODE "MESSAGE(\"\nInstall SUNLINSOL_PSBLAS\n\")")

# Add F90 module if F2003 interface is enabled
IF(F90_FOUND AND F2003_INTERFACE_ENABLE)
  ADD_SUBDIRECTORY(F90)
ENDIF(F90_FOUND AND F2003_INTERFACE_ENABLE)

# Add variable sunlinsolpsblasfcg_SOURCES with the sources for the SUNLINSOLPSBLASFCG lib
SET(sunlinsolpsblasfcg_SOURCES sunlinsol_psblas.c)

# Add variable shared_SOURCES with the common SUNDIALS sources which will
# also be included in the SUNLINSOLPCG library
SET(shared_SOURCES
  ${sundials_SOURCE_DIR}/src/sundials/sundials_math.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_nvector.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_linearsolver.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_iterative.c
  )

# Add variable sunlinsolpcg_HEADERS with the exported SUNLINSOLPCG header files
SET(sunlinsolpcg_HEADERS
  ${sundials_SOURCE_DIR}/include/sunlinsol/sunlinsol_psblas.h
  )

# Add source directory to include directories
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(${MLD2P4_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PSBLAS_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PSBLAS_EXT_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MLD2P4_EXT_INCLUDE_DIR})
INCLUDE_DIRECTORIES(/usr/mpi/gcc/openmpi-1.10.4/include/) # TODO it should find mpi.h directly, but doesn't work

# Define C preprocessor flag -DBUILD_SUNDIALS_LIBRARY
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY)

# Define C preprocessor flags for PSBLAS
ADD_DEFINITIONS(-DIPK4 -DLPK8 -DPtr64Bits -DLowerUnderscore)

# Rules for building and installing the static library:
#  - Add the build target for the SUNLINSOLPCG library
#  - Set the library name and make sure it is not deleted
#  - Install the SUNLINSOLPCG library
IF(BUILD_STATIC_LIBS)
  ADD_LIBRARY(sundials_sunlinsolpsblasfcg_static STATIC ${sunlinsolpsblasfcg_SOURCES} ${shared_SOURCES})
  SET_TARGET_PROPERTIES(sundials_sunlinsolpsblasfcg_static
    PROPERTIES OUTPUT_NAME sundials_sunlinsolpsblasfcg CLEAN_DIRECT_OUTPUT 1)
  INSTALL(TARGETS sundials_sunlinsolpsblasfcg_static DESTINATION ${CMAKE_INSTALL_LIBDIR})
ENDIF(BUILD_STATIC_LIBS)

# Rules for building and installing the shared library:
#  - Add the build target for the SUNLINSOLPCG library
#  - Set the library name and make sure it is not deleted
#  - Set VERSION and SOVERSION for shared libraries
#  - Install the SUNLINSOLPCG library
IF(BUILD_SHARED_LIBS)
  ADD_LIBRARY(sundials_sunlinsolpsblasfcg_shared SHARED ${sunlinsolpsblasfcg_SOURCES} ${shared_SOURCES})

  IF(UNIX)
    TARGET_LINK_LIBRARIES(sundials_sunlinsolpsblasfcg_shared m)
  ENDIF()

  SET_TARGET_PROPERTIES(sundials_sunlinsolpsblasfcg_shared
    PROPERTIES OUTPUT_NAME sundials_sunlinsolpcg CLEAN_DIRECT_OUTPUT 1)
  SET_TARGET_PROPERTIES(sundials_sunlinsolpsblasfcg_shared
    PROPERTIES VERSION ${sunlinsollib_VERSION} SOVERSION ${sunlinsollib_SOVERSION})
  INSTALL(TARGETS sundials_sunlinsolpsblasfcg_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
ENDIF(BUILD_SHARED_LIBS)

# Install the SUNLINSOLPCG header files
INSTALL(FILES ${sunlinsolpcg_HEADERS} DESTINATION include/sunlinsol)

#
MESSAGE(STATUS "Added SUNLINSOL_PSBLAS module")
