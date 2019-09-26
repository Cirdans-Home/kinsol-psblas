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
# Find PSBLAS library.
# 

IF(WIN32)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
endif(WIN32)

### Find include dir
find_path(temp_PSBLAS_INCLUDE_DIR psb_base_cbind.h ${PSBLAS_INCLUDE_DIR})
if (temp_PSBLAS_INCLUDE_DIR)
    set(PSBLAS_INCLUDE_DIR ${temp_PSBLAS_INCLUDE_DIR})
    MESSAGE(STATUS "Setting PSBLAS include dir to ${PSBLAS_INCLUDE_DIR}")
endif()
unset(temp_PSBLAS_INCLUDE_DIR CACHE)

if (PSBLAS_LIBRARY)
    # We have (or were given) PSBLAS_LIBRARY - get path to use for any related libs
    get_filename_component(PSBLAS_LIBRARY_DIR ${PSBLAS_LIBRARY} PATH)
    
    # force CACHE update to show user DIR that will be used
    set(PSBLAS_LIBRARY_DIR ${PSBLAS_LIBRARY_DIR} CACHE PATH "" FORCE)
    MESSAGE(STATUS "Setting PSBLAS library dir to ${PSBLAS_LIBRARY_DIR}")
else ()
    # find library with user provided directory path
    set(PSBLAS_LIBRARY_NAMES libpsb_base.a libpsb_prec.a libpsb_cbind.a libpsb_krylov.a libpsb_util.a)
    find_library(PSBLAS_LIBRARY 
      NAMES ${PSBLAS_LIBRARY_NAMES}
      PATHS ${PSBLAS_LIBRARY_DIR} NO_DEFAULT_PATH
      )
    MESSAGE(STATUS "Setting PSBLAS library dir to ${PSBLAS_LIBRARY_DIR}")
    MESSAGE(STATUS "Library names are : ${PSBLAS_LIBRARY_NAMES}")
endif ()
mark_as_advanced(PSBLAS_LIBRARY)

set(PSBLAS_LIBRARIES ${PSBLAS_LIBRARY})
