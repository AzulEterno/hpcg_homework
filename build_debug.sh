#!/bin/bash

#Serialize build task

if [ ! -d "build_Custom/AVX2_DEBUG" ]; then
    mkdir "build_Custom/AVX2_DEBUG"
fi

 

cd "build_Custom/AVX2_DEBUG"

../../configure MY_MPI_GCC_OMP_DEBUG_PRINT
make