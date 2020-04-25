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
find_path(temp_PSBLAS_EXT_INCLUDE_DIR Make.inc.ext ${PSBLAS_EXT_INCLUDE_DIR})
if (temp_PSBLAS_EXT_INCLUDE_DIR)
    set(PSBLAS_EXT_INCLUDE_DIR ${temp_PSBLAS_EXT_INCLUDE_DIR})
    MESSAGE(STATUS "Setting PSBLAS-EXT include dir to ${PSBLAS_EXT_INCLUDE_DIR}")
endif()
unset(temp_PSBLAS_EXT_INCLUDE_DIR CACHE)

if (PSBLAS_EXT_LIBRARY)
    # We have (or were given) PSBLAS_LIBRARY - get path to use for any related libs
    get_filename_component(PSBLAS_EXT_LIBRARY_DIR ${PSBLAS_EXT_LIBRARY} PATH)

    # force CACHE update to show user DIR that will be used
    set(PSBLAS_EXT_LIBRARY_DIR ${PSBLAS_EXT_LIBRARY_DIR} CACHE PATH "" FORCE)
    MESSAGE(STATUS "Setting PSBLAS library dir to ${PSBLAS_EXT_LIBRARY}")
else ()
    # find library with user provided directory path
    set(PSBLAS_EXT_LIBRARY_NAMES libpsb_ext.a)
    find_library(PSBLAS_EXT_LIBRARY 
      NAMES ${PSBLAS_EXT_LIBRARY_NAMES}
      PATHS ${PSBLAS_EXT_LIBRARY_DIR} NO_DEFAULT_PATH
      )
    MESSAGE(STATUS "Setting PSBLAS library dir to ${PSBLAS_EXT_LIBRARY_DIR}")
    MESSAGE(STATUS "Library names are : ${PSBLAS_EXT_LIBRARY_NAMES}")
endif ()
mark_as_advanced(PSBLAS_EXT_LIBRARY)

set(PSBLAS_EXT_LIBRARIES ${PSBLAS_EXT_LIBRARY})
