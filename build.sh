#!/bin/bash

#Serialize build task

if [ ! -d "build_Custom" ]; then
    mkdir "build_Custom"
fi


if [ ! -d "build_Custom/AVX2" ]; then
    mkdir "build_Custom/AVX2"
fi



if [ ! -d "build_Custom/AVX512" ]; then
    mkdir "build_Custom/AVX512"
fi



cd "build_Custom/AVX2"

make clean && ../../configure MY_MPI_GCC_OMP
make


cd ../../

cd "build_Custom/AVX512"

make clean && ../../configure MY_MPI_GCC_OMP_AVX512
make
