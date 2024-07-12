#!/bin/bash

export OMP_DISPLAY_ENV=TRUE

Input_File_Template="bin/hpcg_data_template.dat"

InnerDirLevel="../../../../"

Executable_File="${InnerDirLevel}bin/xhpcg"
Extract_Program="python ${InnerDirLevel}convert_json_result.py"
Test_Folder="testing"

if [ ! -d "${Test_Folder}/REF" ]; then
    mkdir -p "${Test_Folder}/REF"
fi



testBlkSizeArray=(32 64 128 256)

numbers=(1 2 4)

#block_size=128
test_time=10



# Define an array with the string list
Method_Names=("REF" "ZCY" "LWB")


for block_size in "${testBlkSizeArray[@]}"; do

    # Initialize a counter
    index=0

    # Iterate over the array and print each string with its index
    for m_name in "${Method_Names[@]}"; do
        # Iterate over the array and print each number
        for np_count in "${numbers[@]}"; do 
            echo " $m_name: $np_count"
            test_result_folder="${Test_Folder}/${block_size}/${m_name}/${np_count}"

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

                cd "${InnerDirLevel}"
            
            elif [[ $file_count -ge 2 ]]; then
                cd "${test_result_folder}"
                echo "Multiple result files found: ${file_count}, Matched files: ${matching_files}"
                # Sort files by modification time and delete all but the newest one
                sorted_files=$(ls -t HPCG-Benchmark_*.txt)
                newest_file=$(echo "$sorted_files" | head -n 1)
                echo "Keeping newest file: ${newest_file}"
                old_files=$(echo "$sorted_files" | tail -n +2)
                echo "Deleting older files: ${old_files}"
                for file in $old_files; do
                    rm -f "$file"
                    echo "Deleted: $file"
                done
                # Transcribe the newest file into json
                echo "Transcripting result \"${newest_file}\" into json."
                cat "${newest_file}" | ${Extract_Program} > "Result.json"
                
                cd "${InnerDirLevel}"
            else
                
                cd "${test_result_folder}"
                echo "Result folder existed for Input Size = ${block_size}, m_name = ${m_name}, np_count = ${np_count}, pwd: $(pwd)";

                matching_files=$(ls HPCG-Benchmark_*.txt)
                file_count=$(echo "$matching_files" | wc -w)

                if [[ file_count -eq 1 ]]; then
                    # Print matching files
                    for file in $matching_files; do
                        echo "Transcripting result \"${file}\" into json."

                        cat ${file} | ${Extract_Program} > "Result.json"
                    done
                else
                    echo "Unexpected result file match count: ${file_count}, Matched files: ${matching_files}"
                fi

                cd "${InnerDirLevel}"
            fi


        done

        ((index++))
    done

done