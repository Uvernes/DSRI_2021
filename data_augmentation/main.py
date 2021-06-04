from tslearn.utils import to_time_series
from tslearn.barycenters import dtw_barycenter_averaging_petitjean
from tslearn.metrics import cdist_dtw
from data_extraction_and_storage_methods import *
from smote_based_wDBA import *
from conversion_and_utility_methods import *
from timestamp_methods import create_synthetic_timestamps
import time
import math
from jittering import apply_jittering_to_dataset
import numpy as np
import sys


PROPORTION_OF_DATASET = 0.1
SYNTHETIC_DATA_PATH = r"C:\Users\uvern\Dropbox\My PC (LAPTOP-554U8A6N)\Documents\DSRI\Data\usneedle_data" \
                      r"\usneedle_data\SyntheticData\NoviceData"

# Running main creates synthetic time series using both NeedleTipToReference and ProbeToReference novice data, and it
# writes this data to a directory


def main():

    tic = time.perf_counter()

    # Extract datasets
    novice_set_needle_tip_to_ref, timestamps_set = get_subset_of_dataset("Novice", "NeedleTipToReference")
    novice_set_probe_to_ref, _ = get_subset_of_dataset("Novice", "ProbeToReference")
    size = math.floor(PROPORTION_OF_DATASET * len(novice_set_needle_tip_to_ref))
    novice_set_needle_tip_to_ref = novice_set_needle_tip_to_ref[0:size]
    novice_set_probe_to_ref = novice_set_probe_to_ref[0:size]
    timestamps_set = timestamps_set[0:size]

    # Creating synthetic time series using SMOTE wDBA
    synthetic_needle_tip_to_ref = smote_based_weighted_dba(novice_set_needle_tip_to_ref, N=100)
    synthetic_probe_to_ref = smote_based_weighted_dba(novice_set_probe_to_ref, N=100)

    # Fix all rotation matrices
    for i in range(len(synthetic_needle_tip_to_ref)):
        fix_rotation_matrices(synthetic_needle_tip_to_ref[i])
        fix_rotation_matrices(synthetic_probe_to_ref[i])

    # Create additional synthetic time series by applying jittering
    synthetic_needle_tip_to_ref += apply_jittering_to_dataset(synthetic_needle_tip_to_ref, num_jitters_per_sample=2)
    synthetic_probe_to_ref += apply_jittering_to_dataset(synthetic_probe_to_ref, num_jitters_per_sample=2)

    # Synthetic timestamps (assumption - dif. sequence types for the same trial share the same timestamp values)
    synthetic_timestamps = create_synthetic_timestamps(synthetic_needle_tip_to_ref, timestamps_set)

    # Write synthetic data to MHA files
    write_dataset_to_mha_files(synthetic_needle_tip_to_ref, synthetic_timestamps, "NeedleTipToReference",
                               SYNTHETIC_DATA_PATH, clear_existing_synthetic_directories=True)
    write_dataset_to_mha_files(synthetic_probe_to_ref, synthetic_timestamps, "ProbeToReference",
                               SYNTHETIC_DATA_PATH)

    toc = time.perf_counter()
    print("\nMinutes elapsed: %.2f" % ((toc - tic) / 60))
    # Output: ~ 0.34 minutes for 10% of novice set
    #         ~ 26.88 minutes for entire dataset


main()


# ------ Testing example of creating synthetic time series using novice data -------- #

# tic = time.perf_counter()
#
# novice_set, timestamps_set = get_subset_of_dataset("Novice", "NeedleTipToReference")
# # Only looking at 5 novice examples
# novice_set = novice_set[0:5]
# timestamps_set = timestamps_set[0:5]
# # distances = cdist_dtw(novice_set)
# # print(distances)
# synthetic_samples = smote_based_weighted_dba(novice_set, 100, 1)
#
# print("Synthetic time series before fixing R matrices (only printing values corresponding to first 2 time stamps):")
# for t_series in synthetic_samples:
#     print(t_series[0:2])
#
# print("Check matrix multiplication of R'*R, for one specific example (before fixing)")
# product = np.dot(list_to_matrix(synthetic_samples[1][1])[0:3, 0:3], list_to_matrix(synthetic_samples[1][1])[0:3, 0:3].transpose())
# np.savetxt(sys.stdout, product, '%.5f')
#
# print("\nSynthetic time series after fixing R matrices (only printing values corresponding to first 2 time stamps):")
# for i in range(len(synthetic_samples)):
#     fix_rotation_matrices(synthetic_samples[i])
#     print(synthetic_samples[i][0:2])
#
# print("Check matrix multiplication of R'*R is eye(3), for one specific example (after fixing)")
# product = np.dot(list_to_matrix(synthetic_samples[1][1])[0:3, 0:3], list_to_matrix(synthetic_samples[1][1])[0:3, 0:3].transpose())
# np.savetxt(sys.stdout, product, '%.5f')
#
# print("\nSynthetic timestamps for each of our synthetic time series: ")
# synthetic_timestamps_set = create_synthetic_timestamps(synthetic_samples, timestamps_set)
# for timestamps in synthetic_timestamps_set:
#     print(timestamps)
#
# toc = time.perf_counter()
# print("\nMinutes elapsed: %.2f" % ((toc - tic) / 60))  # Output: ~10.86 minutes for entire novice set

"""
Sample output:

>> Synthetic time series before fixing R matrices (only printing values corresponding to first 2 time stamps):
>> [[0.5698482047821939, -0.8159360892251226, 0.06775899317843137, -58.22159783067481, 0.21013466660592942, 0.22981316363445722, 0.9487319707650737, -159.46164355128514, -0.7899955527576474, -0.5257931867738895, 0.3033068988759845, 11.021177450155587], [0.6276635748424124, -0.7666666207130194, 0.05714968688181164, -58.889633526233084, 0.25222160902468643, 0.2888056260315356, 0.9183120064664108, -153.1247886940687, -0.7225745381457338, -0.5588998982266327, 0.3789582703889484, 12.566055392080806]]
>> [[0.6737557547038815, -0.7348990606162443, 0.06798574103696918, -58.51810020243676, 0.22973997551507358, 0.29721208721226355, 0.9261741123992614, -164.15519380030048, -0.701168173788108, -0.6081571452892337, 0.3693338246395181, 10.170434627856526], [0.6264894446047843, -0.771104305484815, 0.06805138862904295, -58.88103518383009, 0.21588519626638875, 0.26423757655636043, 0.9382262792329732, -161.9744463008446, -0.7421511037210728, -0.5719411350599078, 0.33397238822712194, 10.486730717776531]]
>> [[0.7809079301377841, -0.6174496257213682, 0.08228712042195986, -58.55574382323411, 0.16685992768841976, 0.33476468898385797, 0.9267573907620336, -152.84022534208623, -0.6003838375052056, -0.7102412013389703, 0.3656057346064896, 12.238382188480797], [0.7251542077272957, -0.6804284764605887, 0.054193590219754055, -59.16999163997288, 0.21162088295502232, 0.3008178790118042, 0.9283705952866373, -147.49582349043695, -0.6493920512383091, -0.661933772778964, 0.36667615589398206, 11.174270232760936]]
>> [[0.6507591112085087, -0.7542718788236561, 0.07060108417384246, -58.63166184822979, 0.2135263173894475, 0.2738026223803928, 0.9372740728830653, -166.65403704471177, -0.7265041078864055, -0.5945192718038087, 0.33984968600907706, 9.390633833113366], [0.6480495620466244, -0.756562560482254, 0.07221814180469407, -58.355471088250276, 0.21533350159796272, 0.27567607273688893, 0.936267809154467, -165.51033241659755, -0.7284888199234619, -0.5908386372387974, 0.3421690011884074, 10.247496031535583]]
>> [[0.7690691090349674, -0.6341025373077146, 0.07533391569939472, -59.07716591320001, 0.16118258401414895, 0.30677828064254525, 0.9378852307511061, -158.33362788357485, -0.6178822573882274, -0.7092709476664315, 0.33857687999870456, 10.377249368708368], [0.774559713952656, -0.6264321401217087, 0.07949605036896991, -58.75106461169457, 0.15979066971633407, 0.31608354716808496, 0.9349225789310156, -156.50485590662913, -0.6109252314884563, -0.7116217683730576, 0.34567800781911134, 11.22646762144752]]
>> Check matrix multiplication of R'*R, for one specific example (before fixing)
>> 0.99172 -0.00466 -0.00120
>> -0.00466 0.99670 0.00199
>> -0.00120 0.00199 0.98944

>> Synthetic time series after fixing R matrices (only printing values corresponding to first 2 time stamps):
>> [[0.5713857709455517, -0.8177179900670871, 0.06968205996956058, -58.22159783067481, 0.21180762425069982, 0.2289659767956331, 0.9501116312198776, -159.46164355128514, -0.7928781943465418, -0.5281210753138579, 0.3040268059548411, 11.021177450155587], [0.6325718498685539, -0.7719183928196315, 0.06320482244759518, -58.889633526233084, 0.25845390635805976, 0.28731655765829345, 0.922307309948071, -153.1247886940687, -0.7301057683939596, -0.5670901079388926, 0.38125369038964085, 12.566055392080806]]
>> [[0.6741583506316337, -0.7354118902454416, 0.06840957505540565, -58.51810020243676, 0.230170050588267, 0.2971985287797795, 0.9266578561169859, -164.15519380030048, -0.701806430638698, -0.6089682965284484, 0.3696286619533306, 10.170434627856526], [0.6291635662285049, -0.7740471964037718, 0.07073997929098325, -58.88103518383009, 0.21846197165099815, 0.2634419737951774, 0.9396130551377295, -161.9744463008446, -0.7459407308043893, -0.5757163052948562, 0.33484796840454584, 10.486730717776531]]
>> [[0.7817004929249255, -0.6181757145408474, 0.08248106032806499, -58.55574382323411, 0.1672809635847612, 0.3352396484198272, 0.9271631233766283, -152.84022534208623, -0.600800647954946, -0.710966359316152, 0.36546602761084873, 12.238382188480797], [0.7280462626056814, -0.6833210331437843, 0.05496367135856063, -59.16999163997288, 0.21352162767134725, 0.30222355940873424, 0.929015303778627, -147.49582349043695, -0.6514270135805537, -0.6646301872482382, 0.36593655211784215, 11.174270232760936]]
>> [[0.6516274754267876, -0.7551704162830446, 0.07140921256956756, -58.63166184822979, 0.21431076555007905, 0.27359228248952255, 0.9376663365672852, -166.65403704471177, -0.7276348871778013, -0.5957053846769118, 0.340121398369291, 9.390633833113366], [0.6488515970340906, -0.7574022831460676, 0.07303003841869465, -58.355471088250276, 0.2161255210905304, 0.27546869859667117, 0.9366999280595707, -165.51033241659755, -0.729576153776738, -0.5919955891546679, 0.34243226813745364, 10.247496031535583]]
>> [[0.7693783647192166, -0.6343382749753107, 0.0753119167418476, -59.07716591320001, 0.1612848462447243, 0.3069780742761673, 0.93795077711227, -158.33362788357485, -0.618097185136609, -0.7094923641696523, 0.3384914399377885, 10.377249368708368], [0.7750845643006481, -0.6268365498074535, 0.07949753460556096, -58.75106461169457, 0.15998853343276723, 0.316410274224326, 0.9350338002100882, -156.50485590662913, -0.6112671980017469, -0.7120115716691511, 0.3455313798431605, 11.22646762144752]]
>> Check matrix multiplication of R'*R is eye(3), for one specific example (after fixing)
>> 1.00000 0.00000 0.00000
>> 0.00000 1.00000 0.00000
>> 0.00000 0.00000 1.00000

>> Minutes elapsed: 0.03

"""

# Potential idea - needs to be tweaked
# def generate_and_store_synthetic_samples(skill_level, sequence_type, proportion_of_dataset, shuffleIndices=None):
#
#     # Put in a better location afterwards
#     # - shuffle indices lets you get a random subset of dataset. But must be passed in. Lets us reuse shuffling
#     # for multiple sequence types
#
#     novice_set, timestamps_set = get_subset_of_dataset("Novice", "NeedleTipToReference")


# ------- Code for generating synthetic time series using novice data ------- #
