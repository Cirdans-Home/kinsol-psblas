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
# Find MLD2P4 library.
# 

IF(WIN32)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
endif(WIN32)

### Find include dir
find_path(temp_MLD2P4_EXT_INCLUDE_DIR psb_base_cbind.h ${MLD2P4_EXT_INCLUDE_DIR})
if (temp_MLD2P4_EXT_INCLUDE_DIR)
    set(MLD2P4_EXT_INCLUDE_DIR ${temp_MLD2P4_EXT_INCLUDE_DIR})
    MESSAGE(STATUS "Setting MLD2P4-EXT include dir to ${MLD2P4_EXT_INCLUDE_DIR}")
endif()
unset(temp_MLD2P4_EXT_INCLUDE_DIR CACHE)

if (MLD2P4_EXT_LIBRARY)
    # We have (or were given) MLD2P4_LIBRARY - get path to use for any related libs
    get_filename_component(MLD2P4_EXT_LIBRARY_DIR ${MLD2P4_EXT_LIBRARY} PATH)
    
    # force CACHE update to show user DIR that will be used
    set(MLD2P4_EXT_LIBRARY_DIR ${MLD2P4_EXT_LIBRARY_DIR} CACHE PATH "" FORCE)
    MESSAGE(STATUS "Setting MLD2P4-EXT library dir to ${MLD2P4_EXT_LIBRARY_DIR}")
else ()
    # find library with user provided directory path
    set(MLD2P4_EXT_LIBRARY_NAMES libmld_parmatch.a)
    find_library(MLD2P4_LIBRARY 
      NAMES ${MLD2P4_EXT_LIBRARY_NAMES}
      PATHS ${MLD2P4_EXT_LIBRARY_DIR} NO_DEFAULT_PATH
      )
    MESSAGE(STATUS "Setting MLD2P4-EXT library dir to ${MLD2P4_EXT_LIBRARY_DIR}")
    MESSAGE(STATUS "Library names are : ${MLD2P4_EXT_LIBRARY_NAMES}")
endif ()
mark_as_advanced(MLD2P4_EXT_LIBRARY)

set(MLD2P4_EXT_LIBRARIES ${MLD2P4_EXT_LIBRARY})
