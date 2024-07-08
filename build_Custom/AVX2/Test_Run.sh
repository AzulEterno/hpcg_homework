#!/bin/bash

export OMP_DISPLAY_ENV=TRUE

Input_File_Template="bin/hpcg_data_template.dat"
Executable_File="../../../bin/xhpcg"
Extract_Program="python ../../../convert_json_result.py"
Test_Folder="testing"

if [ ! -d "${Test_Folder}/REF" ]; then
    mkdir -p "${Test_Folder}/REF"
fi





numbers=(1 2 4)

block_size=128
test_time=10



# Define an array with the string list
Method_Names=("REF" "ZCY" "LWB")

# Initialize a counter
index=0

# Iterate over the array and print each string with its index
for m_name in "${Method_Names[@]}"; do
    

    # Iterate over the array and print each number
    for np_count in "${numbers[@]}"; do 
        echo " $m_name: $np_count"
        test_result_folder="${Test_Folder}/${m_name}/${np_count}"

        if [ ! -d "${test_result_folder}" ]; then
            mkdir -p "${test_result_folder}"
            cp "${Input_File_Template}" "${test_result_folder}/hpcg.dat"


            #Inject test input parameter
            echo "${block_size} ${block_size} ${block_size}" >> "${test_result_folder}/hpcg.dat"
            echo "${test_time}" >> "${test_result_folder}/hpcg.dat"

            cd "${test_result_folder}"
            

            mpirun -np $np_count ${Executable_File} --mt=${index} --dt=1 --wt=1


            # Error stop
            if [[ $? != 0 ]]; then
                exit 1;

            fi

            cd "../../../"
        else
            cd "${test_result_folder}"
            matching_files=$(ls HPCG-Benchmark_*.txt)
            file_count=$(echo "$matching_files" | wc -w)

            if [[ file_count == 1 ]]; then
                # Print matching files
                for file in $matching_files; do
                    cat ${file} | ${Extract_Program} > "Result.json"
                done
            fi

            cd "../../../"
        fi


    done

    ((index++))
done