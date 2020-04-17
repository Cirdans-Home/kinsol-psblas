#!/bin/bash
# This script reads as input the Make.inc file for the PSBLAS/MLD2P4 library
# and produces a file that can be included from the CMake installer of KINSOL

liblocation=$1
wheretoputfile=$2

echo "The Make.inc files are in "$liblocation

cat <<EOF > $liblocation/Makefile
include $liblocation/Make.inc.psblas
include $liblocation/Make.inc.mld2p4
include $liblocation/Make.inc.ext
all:
	@echo "" > $wheretoputfile/makeincinputcmake
	@echo "SET(PSBDEFINES "\${MLDFDEFINES}")\n" >> $wheretoputfile/makeincinputcmake
	@echo "SET(PSBCDEFINES "\${MLDCDEFINES}")\n" >> $wheretoputfile/makeincinputcmake
	@echo "SET(PSBLDLIBS "\${MLDLDLIBS}")\n" >> $wheretoputfile/makeincinputcmake
	@echo "SET(PSBLAS_LIBS "\${PSBLAS_LIBS} "-lpsb_cbind)\n" >> $wheretoputfile/makeincinputcmake
	@echo "SET(PSBRSBLDLIBS "\${LIBRSB_LIBS}")\n" >> $wheretoputfile/makeincinputcmake
	@echo "SET(PSBGPULDLIBS "\${SPGPU_LIBS} \${CUDA_LIBS}")\n" >> $wheretoputfile/makeincinputcmake
EOF

echo "Creating input file for CMake : makeinputcmake"

(cd $liblocation ; make all)

(cd $liblocation ; rm Makefile)
