import sys
import math
import random
import numpy as np
from data_augmentation import jittering, smote_based_wDBA
from enum import Enum
from utility.conversion_methods import join_dataset_dictionaries, fix_rotation_matrices, list_to_matrix

DATA_AUGMENTATION_TECHNIQUES = ["smote_based_wdba", "jittering"]


class ProficiencyLabel(Enum):
    """
        Scanned region
    """
    Novice = 0
    Expert = 1


# Represents the data for a single surgery. Stores time series data captured from the surgery, as well as related info
# May store multiple time series if multiple sequence types are considered (assumption is they all share the same timestamps)

class SurgeryData:

    def __init__(self, skill_level, surgery_type, timestamps=[]):
        self.sequences = dict()
        self.skill_level = skill_level
        self.surgery_type = surgery_type
        self.timestamps = timestamps

    def add_sequence(self, sequence_type, time_series):
        self.sequences[sequence_type] = time_series


# Instances stores all surgeries data recorded for some participant
class ParticipantData:

    def __init__(self, name):
        self.name = name
        self.surgeries = []
        self.surgeries_stats = dict(
            novices_IP=0,
            novices_OOP=0,
            experts_IP=0,
            experts_OOP=0
        )

    # To add a surgery, must pass in a Surgery object
    def add_surgery(self, surgery):

        self.surgeries.append(surgery)
        self.update_surgery_stats(surgery)

    def update_surgery_stats(self, surgery):

        skill_level = surgery.skill_level
        surgery_type = surgery.surgery_type
        for stat in self.surgeries_stats:
            if (skill_level.lower() in stat.lower()) and (surgery_type.lower() in stat.lower()):
                self.surgeries_stats[stat] += 1
                break


# Contains a dictionary that stores participant names as keys, and the corresponding ParticipantsStorage objects as keys
# Altogether, it contains all time series data
class ParticipantsStorage:

    def __init__(self):
        self.participants = dict()
        self.surgeries_stats = dict(
            surgeries=0,
            novices=0,
            experts=0,
            novices_IP=0,
            novices_OOP=0,
            experts_IP=0,
            experts_OOP=0
        )

    # Adds a surgery to the storage, within the corresponding participant's storage. If the participant does not exist,
    # then they are created
    def add_surgery(self, participant_name, surgery):

        if participant_name not in self.participants:
            self.participants[participant_name] = ParticipantData(participant_name)
        self.participants[participant_name].add_surgery(surgery)
        self.update_surgeries_stats(surgery)

    def add_participant(self, participant):

        if (not isinstance(participant, ParticipantData)) or participant.name in self.participants:
            return False
        self.participants[participant.name] = participant
        for surgery in participant.surgeries:
            self.update_surgeries_stats(surgery)
        return True

    def pop_participant(self, participant_name=None):

        if len(self.participants) == 0:
            return None
        participant = None
        names = list(self.participants.keys())
        if participant_name is None:
            # Pops random participant
            participant = self.participants.pop(names[random.randint(0, len(self.participants)-1)])
        elif participant_name in names:
            participant = self.participants.pop(participant_name)
        else:
            return None

        for surgery in participant.surgeries:
            self.update_surgeries_stats(surgery, False)

        return participant

    def update_surgeries_stats(self, surgery, add=True):

        # Values we add below depend on whether a surgery is being added or removed
        term = 1 if add else -1
        skill_level = surgery.skill_level
        surgery_type = surgery.surgery_type
        self.surgeries_stats["surgeries"] += term
        self.surgeries_stats[(skill_level + "s").lower()] += term  # Updates number of novices / experts
        for stat in self.surgeries_stats:
            if (skill_level.lower() in stat.lower()) and (surgery_type.lower() in stat.lower()):
                self.surgeries_stats[stat] += term
                break

    # Note: In all ratios, we return math.inf if denominator is 0
    def novices_to_experts_ratio(self):

        return self.surgeries_stats["novices"] / self.surgeries_stats["experts"] if  \
            self.surgeries_stats["experts"] else math.inf

    def novices_IP_to_OOP_ratio(self):

        return self.surgeries_stats["novices_IP"] / self.surgeries_stats["novices_OOP"] if \
            self.surgeries_stats["novices_OOP"] else math.inf

    def experts_IP_to_OOP_ratio(self):

        return self.surgeries_stats["experts_IP"] / self.surgeries_stats["experts_OOP"] if \
            self.surgeries_stats["experts_OOP"] else 0


class Jittering:

    def __init__(self, rotation_magnitude=1, translation_magnitude=1):

        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude

    def execute(self, dataset, synthetic_amount):
        return jittering.jittering_specified_amount(
            dataset, synthetic_amount, self.rotation_magnitude, self.translation_magnitude)


class SmoteBasedWDBA:

    def __init__(self, k=None):
        self.k = k

    def execute(self, dataset, synthetic_amount):
        return smote_based_wDBA.smote_based_weighted_dba_specified_amount(
            dataset, synthetic_amount, self.k)


class DataAugmentationInstruction:

    # Technique refers to the data augmentation technique to use
    def __init__(self, technique, augment_synthetic=False):
        self.technique = technique
        self.augment_synthetic = augment_synthetic


class BalanceDataset(DataAugmentationInstruction):

    def __init__(self, technique, augment_synthetic=False):

        super().__init__(technique, augment_synthetic)

    # Here is the code for balancing a dataset. First balance novices IP, OOP and experts IP, OOP.
    # Afterwards, balance novices and experts . We always up-sample, via the specified technique.
    # Returns synthetic samples
    def execute(self, original_dataset, synthetic_dataset):

        enhanced_set = join_dataset_dictionaries(original_dataset, synthetic_dataset)
        # print("Num of novices IP:", len(enhanced_set["Novices"]["IP"]))
        # print("Num of novices OOP:", len(enhanced_set["Novices"]["OOP"]))
        # print("Num of experts OP:", len(enhanced_set["Experts"]["IP"]))
        # print("Num of experts OOP:", len(enhanced_set["Experts"]["OOP"]))

        # If enhanced_set assigned, then new samples may be creating using the synthetic ones passed in.
        set_used_for_augmentation = enhanced_set if self.augment_synthetic else original_dataset

        # Balance novices IP and OOP, and then experts IP, OOP (in enhanced set)
        for skill_level in ["Novices", "Experts"]:
            desired_size = max(len(enhanced_set[skill_level]["IP"]), len(enhanced_set[skill_level]["OOP"]))
            for surgery_type in ["IP", "OOP"]:
                synthetic_amount = desired_size - len(enhanced_set[skill_level][surgery_type])
                synthetic_dataset[skill_level][surgery_type] += \
                    self.technique.execute(set_used_for_augmentation[skill_level][surgery_type], synthetic_amount)

        # Balance novices and experts. At this point, 1:1 IP to OOP ratio for both skill levels
        num_novices_IP = len(original_dataset["Novices"]["IP"]) + len(synthetic_dataset["Novices"]["IP"])
        num_experts_IP = len(original_dataset["Experts"]["IP"]) + len(synthetic_dataset["Experts"]["IP"])
        desired_size = max(num_novices_IP, num_experts_IP)

        # print("num_novices_IP:", num_novices_IP)
        # print("num_experts_IP", num_experts_IP)

        for skill_level in ["Novices", "Experts"]:
            for surgery_type in ["IP", "OOP"]:
                # Want to get to the desired size. Need to account both original and synthetic samples
                synthetic_amount = desired_size - (len(original_dataset[skill_level][surgery_type]) +
                                                   len(synthetic_dataset[skill_level][surgery_type]))
                synthetic_dataset[skill_level][surgery_type] += \
                    self.technique.execute(set_used_for_augmentation[skill_level][surgery_type], synthetic_amount)

        return synthetic_dataset


# Novices IP, Novices OOP, Experts IP, Experts OOP, which make up the dataset, are all increased by the given factor
# independently.
# NOTE: ** Increase is relative to the size of the enhanced set, not the original dataset
class IncreaseDatasetProportionally(DataAugmentationInstruction):

    def __init__(self, technique, increase_factor, augment_synthetic=False):

        super().__init__(technique, augment_synthetic)
        self.increase_factor = increase_factor

    def execute(self, original_dataset, synthetic_dataset):

        enhanced_set = join_dataset_dictionaries(original_dataset, synthetic_dataset)
        set_used_for_augmentation = enhanced_set if self.augment_synthetic else original_dataset

        if self.increase_factor <= 1:
            return synthetic_dataset

        # Increase each division separately
        for skill_level in ["Novices", "Experts"]:
            for surgery_type in ["IP", "OOP"]:
                desired_size = round(self.increase_factor * len(enhanced_set[skill_level][surgery_type]))
                synthetic_amount = desired_size - len(enhanced_set[skill_level][surgery_type])
                synthetic_dataset[skill_level][surgery_type] += \
                    self.technique.execute(set_used_for_augmentation[skill_level][surgery_type], synthetic_amount)

        return synthetic_dataset


# Novices IP, Novices OOP, Experts IP, Experts OOP, which make up the dataset, are all increased by aprx. the same number
# of synthetic samples such that the dataset reaches the required size, which is len(original_dataset) * increase_factor.
# -This means increase is relative to size of the original dataset
# NOTE: the resulting enhanced set may be slightly imbalanced
#  e.g perhaps 4 Novices IP are required and 3 for each of the other types
# ** NOTE: This method only works well for increase_factor >= 2 . If it's less, increase_factor may already be exceeded
# when samples are first balanced

class IncreaseDatasetBalanced(DataAugmentationInstruction):

    def __init__(self, technique, increase_factor, augment_synthetic=False):

        super().__init__(technique, augment_synthetic)
        self.increase_factor = increase_factor
        self.balance_instruction = BalanceDataset(technique, augment_synthetic)

    def execute(self, original_dataset, synthetic_dataset):

        if self.increase_factor < 2:
            print("Note: Increase factor must be >= 2. No data augmentation performed... (otherwise factor is overshot when balancing)")
            return synthetic_dataset

        # First balance dataset
        synthetic_dataset = self.balance_instruction.execute(original_dataset, synthetic_dataset)

        enhanced_set = join_dataset_dictionaries(original_dataset, synthetic_dataset)
        set_used_for_augmentation = enhanced_set if self.augment_synthetic else original_dataset

        # Find number of surgeries in original and enhanced datasets
        original_size = 0
        cur_size = 0
        for skill_level in ["Novices", "Experts"]:
            for surgery_type in ["IP", "OOP"]:
                original_size += len(original_dataset[skill_level][surgery_type])
                cur_size += len(enhanced_set[skill_level][surgery_type])

        desired_size = round(self.increase_factor * original_size)

        synthetic_samples_required = desired_size - cur_size

        if synthetic_samples_required <= 0:
            return synthetic_dataset

        # Increase each division separately, all by aprx. the same amount
        # After attempting to split data augmentation evenly, may require 1 more of a certain division(s)
        # then others. This occurs if desired_size is not a multiple of 4
        extra_samples_required = synthetic_samples_required % 4
        for skill_level in ["Novices", "Experts"]:
            for surgery_type in ["IP", "OOP"]:
                synthetic_amount = synthetic_samples_required // 4
                if extra_samples_required > 0:
                    synthetic_amount += 1
                    extra_samples_required -= 1

                synthetic_dataset[skill_level][surgery_type] += \
                    self.technique.execute(set_used_for_augmentation[skill_level][surgery_type], synthetic_amount)

        return synthetic_dataset


class DataAugmentationController:

    # Instructions stores list of data augmentation steps to perform
    def __init__(self, instructions=None):
        if instructions is None:
            self.instructions = []
        else:
            self.instructions = instructions

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def execute(self, original_dataset):
        synthetic_dataset = dict(
            Novices=dict(IP=[], OOP=[]),
            Experts=dict(IP=[], OOP=[])
        )
        for instruction in self.instructions:
            synthetic_dataset = instruction.execute(original_dataset, synthetic_dataset)

        # print("Matrix multiplication before:")
        # product = np.dot(list_to_matrix(synthetic_dataset["Novices"]["IP"][0][0])[0:3, 0:3],
        #                  list_to_matrix(synthetic_dataset["Novices"]["IP"][0][0])[0:3, 0:3].transpose())
        # np.savetxt(sys.stdout, product, '%.5f')

        # Fix all rotation matrices, for all time series
        for skill_level in synthetic_dataset:
            for surgery_type in synthetic_dataset[skill_level]:
                for i in range(len(synthetic_dataset[skill_level][surgery_type])):
                    fix_rotation_matrices(synthetic_dataset[skill_level][surgery_type][i])

        # print("Matrix multiplication after:")
        # product = np.dot(list_to_matrix(synthetic_dataset["Novices"]["IP"][0][0])[0:3, 0:3],
        #                  list_to_matrix(synthetic_dataset["Novices"]["IP"][0][0])[0:3, 0:3].transpose())
        # np.savetxt(sys.stdout, product, '%.5f')

        return synthetic_dataset


class Transcript(object):
    """
    Transcript - direct print output to a file, in addition to terminal (from Stack Overflow, but modified).

    Usage:
        import transcript
        transcript.start('logfile.log')
        print("inside file")
        transcript.stop()
        print("outside file")
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):

        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass





