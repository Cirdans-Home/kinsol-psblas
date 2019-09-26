# ---------------------------------------------------------------------------
# Programmer: F. Durastante @ IAC-CNR
# ---------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------------------
# PSBLAS tests for SUNDIALS CMake-based configuration.
# ---------------------------------------------------------------------------

### This is only set if running GUI - simply return first time enabled
IF(MLD2P4_EXT_DISABLED)
  SET(MLD2P4_EXT_DISABLED FALSE CACHE INTERNAL "GUI - now enabled" FORCE)
  RETURN()
ENDIF()

SET(MLD2P4_EXT_FOUND FALSE)

# set PSBLAS_LIBRARIES
include(FindMLD2P4_EXT)

SET(MLD2P4_EXT_FOUND TRUE)
