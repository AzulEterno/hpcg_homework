import sys
import json
from collections import defaultdict
import traceback
import pandas as pd
import os


def main():
    dict_data_key_list: list[str] = []
    dict_data_list: list[dict] = []

    Methods = ["REF", "ZCY", "LMB"]
    block_sizes = [32, 64, 128, 256]
    base_path = "testing"

    for block_size in block_sizes:
        block_size_folder_path = os.path.join(base_path, str(block_size))
        for m_index, method_name in enumerate(Methods):
            test_batch_folder_path = os.path.join(block_size_folder_path, method_name)
            # print(os.listdir(test_batch_folder_path))

            for np_num in os.listdir(test_batch_folder_path):
                single_bench_result_path = os.path.join(
                    test_batch_folder_path, f"{np_num}", "Result.json"
                )
                if os.path.isfile(single_bench_result_path):
                    with open(single_bench_result_path, mode="r") as reader:
                        raw_data = json.loads(reader.read())

                    new_data_bundle = {}

                    for item_name, value in raw_data.get("GFLOP/s Summary", {}).items():
                        new_data_bundle[f"{item_name} - GFLOP/s"] = value

                    for item_name, value in raw_data.get(
                        "User Optimization Overheads", {}
                    ).items():
                        new_data_bundle[item_name] = value
                    for item_name, value in raw_data.get(
                        "DDOT Timing Variations", {}
                    ).items():
                        new_data_bundle[item_name] = value

                    for item_name, value in raw_data.get(
                        "Benchmark Time Summary", {}
                    ).items():
                        new_data_bundle[f"{item_name} - Execution Time (s)"] = value

                    new_data_bundle["Result - GFLOP/s"] = raw_data.get(
                        "Final Summary", {}
                    ).get("HPCG result is VALID with a GFLOP/s rating of")

                    new_data_key = f"I{block_size}-{method_name}-{np_num}"
                    new_data_bundle["Id"] = new_data_key

                    dict_data_key_list.append(new_data_key)
                    dict_data_list.append(new_data_bundle)
                else:
                    print(f"'{single_bench_result_path}' doesn't exist.")

    df_obj = pd.DataFrame.from_dict(dict_data_list)

    df_obj.set_index("Id", inplace=True)

    df_obj.to_csv("Merged_Results.csv", index=True)


if __name__ == "__main__":
    main()
