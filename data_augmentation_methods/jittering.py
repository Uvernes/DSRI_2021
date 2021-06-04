import sys
import numpy as np
from scipy.spatial.transform import Rotation
from conversion_and_utility_methods import *


def generate_random_transform_matrix(rotation_magnitude, translation_magnitude):
    random_rotation_axis = np.random.normal(0, 1, 3)
    random_rotation_axis = random_rotation_axis / np.linalg.norm(random_rotation_axis)
    random_rotation_magnitude = np.random.normal(0, rotation_magnitude, 1)
    random_rotation_vector = random_rotation_magnitude * random_rotation_axis

    random_rotation = Rotation.from_rotvec(random_rotation_vector)
    random_rotation_matrix = random_rotation.as_matrix()

    random_translation = np.random.normal(0, translation_magnitude, 3)

    random_transform_matrix = np.identity(4)
    random_transform_matrix[0:3, 0:3] = random_rotation_matrix
    random_transform_matrix[0:3, 3] = random_translation

    return random_transform_matrix


def apply_jittering_to_single_time_series(time_series, rotation_magnitude=1, translation_magnitude=1):
    """
    Parameters
    ----------
    time_series - Time series to apply jittering to (applied once)
    rotation_magnitude - Magnitude of rotation matrix
    translation_magnitude - Magnitude of translation vector

    Return
    ------
    A single synthetic time series obtained via applying jittering
    """
    synthetic = []
    for translation_list in time_series:
        translation_matrix = list_to_matrix(translation_list)
        new_translation_matrix = np.dot(translation_matrix,
                                        generate_random_transform_matrix(rotation_magnitude, translation_magnitude))
        synthetic.append(matrix_to_list(new_translation_matrix))
    return synthetic


def apply_jittering_to_dataset(dataset, num_jitters_per_sample, rotation_magnitude=1, translation_magnitude=1):

    """
    Parameters
    ----------
     dataset - Dataset to apply jittering to
     num_jitters_per_sample - number of synthetic samples to create per sample in original dataset
     rotation_magnitude - Magnitude of rotation matrix
     translation_magnitude - Magnitude of translation vector

     Return
     ------
     List of synthetic time series obtained via jittering
    """

    synthetic_samples = []
    for t_series in dataset:
        for i in range(num_jitters_per_sample):
            synthetic_samples.append(apply_jittering_to_single_time_series(t_series))
    return synthetic_samples


# ----------- Testing ------------ #
# random_transform_matrix = generate_random_transform_matrix(1, 1)
# print("Random transform matrix: ")
# print(random_transform_matrix)
# print("\nDouble check R*R' = I :")
# np.savetxt(sys.stdout, np.dot(random_transform_matrix[0:3, 0:3], random_transform_matrix.transpose()[0:3, 0:3]), '%.5f')
#
# print("\nApplying jittering to a single time series...")
# t_1 = [[-0.2195268905925537, 0.9755702249011461, 0.008407175095954811, -58.43038091865748,
#         -0.25417037610269866, -0.0655100326630646, 0.9649382651404776, -156.23547764833717,
#         0.9419157946539483, 0.20969304210396036, 0.26234226475086037, 7.922358839781506],
#        [-0.2195268905925537, 0.9755702249011461, 0.008407175095954811, -58.43038091865748,
#         -0.25417037610269866, -0.0655100326630646, 0.9649382651404776, -156.23547764833717,
#         0.9419157946539483, 0.20969304210396036, 0.26234226475086037, 7.922358839781506]]
#
# print("Original time series:")
# print("Double check R*R' = I for first translation matrix:")
# np.savetxt(sys.stdout, np.dot(list_to_matrix(t_1[0])[0:3, 0:3], list_to_matrix(t_1[0])[0:3, 0:3].transpose()), '%.5f')
# print(t_1)
#
# print("\nSynthetic time series:")
# s_1 = apply_jittering_to_single_time_series(t_1)
# print(s_1)
# print("Double check R*R' = I for first translation matrix:")
# np.savetxt(sys.stdout, np.dot(list_to_matrix(s_1[0])[0:3, 0:3], list_to_matrix(s_1[0])[0:3, 0:3].transpose()), '%.5f')
#
# print("\nApplying jittering to a dataset and printing results...")
# dataset = [t_1, t_1, t_1]
# synthetic_samples = apply_jittering_to_dataset(dataset, 2)
# for synthetic in synthetic_samples:
#     print(synthetic)
# print("Check R*R' = I, for one specific sample's first translation matrix:")
# product = np.dot(list_to_matrix(synthetic_samples[1][0])[0:3, 0:3], list_to_matrix(synthetic_samples[1][0])[0:3, 0:3].transpose())
# np.savetxt(sys.stdout, product, '%.5f')


