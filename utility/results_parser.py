import os
from model_related.classes import Results


# From stack overflow - https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_table(results, file_iterator, result_type):
    for line in file_iterator:
        if line == "\n":
            return
        if "---" in line:
            continue
        # print(line)

        performance_measure = ""
        elements = line.split()
        for ele in elements:
            if isfloat(ele):
                break
            performance_measure += " " + ele
        performance_measure = performance_measure.strip()

        sets = ["training", "validation", "test"]
        set_index = 0
        for ele in elements:
            if isfloat(ele):
                if result_type == "averages":
                    results.averages[sets[set_index]][performance_measure] = float(ele)
                    set_index += 1
                elif result_type == "stddevs":
                    results.stddevs[sets[set_index]][performance_measure] = float(ele)
                    set_index += 1


def read_single_results_file(filename):

    results = Results()
    f = open(filename)
    iter_f = iter(f)
    for line in iter_f:
        if "Performance measure" in line and "mean" in line:
            # print(line)
            read_table(results, iter_f, "averages")
        elif "Performance measure" in line and "stdev" in line:
            # print(line)
            read_table(results, iter_f, "stddevs")

    return results


def read_all_result_files(directory):

    result_objects = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith(".out"):
            continue
        result_objects.append(read_single_results_file(directory + "\\" + filename))
    return result_objects


# Run this function to parse results
def parse_results(directory, output_filename):

    all_results = read_all_result_files(directory)
    averaged_results = Results.create_averaged_results_object(all_results)
    f = open(directory + "\\" + output_filename, "w")
    f.write("\n=> Summary Statistics\n")
    f.write("   ------------------\n\n\n")
    f.write("Average results across all trials:\n\n\n")
    f.write(str(averaged_results))
    f.write("\n\n\nIndividual results\n")
    f.write("------------------\n\n\n")

    for i in range(len(all_results)):
        f.write(str(i + 1) + ".\n")
        f.write(str(all_results[i]) + "\n\n")
    f.close()


# Code to execute for parsing results
directory = \
    r"C:\Users\uvern\PycharmProjects\DSRI\USNeedleClassificationWithDataAugmentation\Results_ComputeCanada\SMOTE_10-x"
output_filename = "SMOTE_10-x_summary.txt"
parse_results(directory, output_filename)


# ------------- TESTING ----------- #

# directory = r"C:\Users\uvern\PycharmProjects\DSRI\USNeedleClassificationWithDataAugmentation\Results_ComputeCanada\SMOTE_5-x"
# single_results = read_single_results_file(
#     r"C:\Users\uvern\PycharmProjects\DSRI\USNeedleClassificationWithDataAugmentation\Results_ComputeCanada\SMOTE_5-x\SMOTE_5-x-50184502.out")
#
# all_results = read_all_result_files(directory)
# averaged_results = Results.create_averaged_results_object(all_results)
#
# # LATER - Create A toString method
# print('Printing single results...')
# print("\nAverages for train, val, test sets:")
# print(single_results.averages)
# print("\nStddevs for train, val, test sets:")
# print(single_results.stddevs)
#
# print("\nReading in all result files...")
# print("Printing info for one of the results objects...")
# print("\nAverages for train, val, test sets:")
# print(all_results[0].averages)
# print("\nStddevs for train, val, test sets:")
# print(all_results[0].stddevs)
#
# print("\nAveraging results from all of the result files...")
# print("\nAverages for train, val, test sets:")
# print(averaged_results.averages)
# print("\nStddevs for train, val, test sets:")
# print(averaged_results.stddevs)
#
# print("\nWriting averaged results object to console...")
# print(averaged_results)

"""
Next steps:

1) Read in multiple result files (into multiple result objects)
2) Create a result object that is the average of these other result objects
3) Create a method in Results that appends individual output to a file in tabular format
4) Print analysis of all results to a text file. This includes both the average of all results, and all of the individual
   results.
5) Run the completed code on previously computed analyses to double check
6) Run this code on the remaining results to analyse
7) Continue to gather more results... according to work meeting notes

DONE ALL THE ABOVE... continue with below

8) Read about the various hypothesis testing methods, starting with confidence intervals 
  Our hope: To get non-overlapping 95% CI intervals

"""


