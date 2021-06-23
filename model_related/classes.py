import sys
import math
import random
from enum import Enum


class ProficiencyLabel(Enum):
    """
        Scanned region
    """
    Novice = 0
    Expert = 1


# Represents the data for a single surgery. Stores the time series captured from the surgery, as well as related info

class SurgeryData:

    def __init__(self, time_series, skill_level, surgery_type, sequence_type="", timestamps=[]):
        self.time_series = time_series
        self.skill_level = skill_level
        self.surgery_type = surgery_type
        self.sequence_type = sequence_type
        self.timestamps = timestamps


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
    def add_surgery(self, participant_name, time_series, skill_level, surgery_type, sequence_type="", timestamps=[]):

        if participant_name not in self.participants:
            self.participants[participant_name] = ParticipantData(participant_name)
        surgery = SurgeryData(time_series, skill_level, surgery_type, sequence_type, timestamps)
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


# class ParticipantsData:
#     def __init__(self):
#         self.store = dict()
#
#     def add_participant(self, part: ParticipantScan) -> bool:
#         if part.get_name() not in self.store:
#             self.store[part.get_name()] = part
#             return True
#
#         return False
#
#     def __getitem__(self, item: str) -> ParticipantScan:
#         return self.store[item]
#
#     def __contains__(self, part_name: str) -> bool:
#         return part_name in self.store
#
#     def __iter__(self) -> ParticipantScan:
#         for p in self.store:
#             yield self.store[p]


class Scan(Enum):
    """
        Scanned region
    """
    LUQ = 'Scan01'
    RUQ = 'Scan02'
    PERICARD = 'Scan03'
    PELVIC = 'Scan04'
    ALL = 'ALL'


TRANSFORM_KEY: str = 'transforms'
TIME_KEY: str = 'time'
PATH_LENGTH_KEY: str = 'path_len'
ANGULAR_SPEED: str = 'ang_speed'
LINEAR_SPEED: str = 'lin_speed'


class TransformationRecord:
    def __init__(self, trans_mat, time_stamp, linear_speed=0, angular_speed=0, path_length=0):
        self.trans_mat = trans_mat
        self.time_stamp = time_stamp
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.path_length = path_length


class RegionScan:
    def __init__(self, reg: Scan):
        self._region = reg
        self.path_len = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.time = 0.0
        self.transformations = []

    def get_regon(self) -> Scan:
        return self._region


class ParticipantScan:
    def __init__(self, part_name):
        self.store = dict()
        self.name = part_name
        self.time = 0.0
        self.path_length = 0.0
        self.angular_speed = 0.0
        self.linear_speed = 0.0

        self.store[Scan.ALL] = RegionScan(Scan.ALL)
        self.store[Scan.LUQ] = RegionScan(Scan.LUQ)
        self.store[Scan.RUQ] = RegionScan(Scan.RUQ)
        self.store[Scan.PELVIC] = RegionScan(Scan.PELVIC)
        self.store[Scan.PERICARD] = RegionScan(Scan.PERICARD)

    def get_transforms(self, reg: Scan) -> list:
        return self.store[reg].transformations

    def get_time(self) -> float:
        return self.time

    def get_reg_time(self, reg: Scan) -> float:
        return self.store[reg].time

    def get_region(self, reg: Scan) -> RegionScan:
        return self.store[reg]

    def add_transform(self, reg, transform_rec: TransformationRecord):
        self.store[reg].transformations.append(transform_rec)

    def get_name(self):
        return self.name

    def set_reg_time(self, reg: Scan, t: float) -> bool:
        if reg not in self.store:
            return False

        self.store[reg][TIME_KEY] = t
        return True

    def set_reg_lin_speed(self, reg: Scan, lin_s: float) -> bool:
        if reg not in self.store:
            return False

        self.store[reg].linear_speed = lin_s
        return True

    def add_reg_time(self, reg: Scan, t: float):
        if reg not in self.store:
            return False

        self.store[reg].time = self.store[reg].time + t
        return True

    def set_time(self, t: float) -> bool:
        self.time = t

    def add_time(self, t: float):
        self.time = self.time + t

    def set_reg_angular_speed(self, reg: Scan, ang_s: float):
        self.store[reg].angular_speed = ang_s

    def set_angular_speed(self, ang_s: float):
        self.angular_speed = ang_s

# --------------- OLD -----------------
# class ParticipantsData:
#     def __init__(self):
#         self.store = dict()
#
#     def add_participant(self, part: ParticipantScan) -> bool:
#         if part.get_name() not in self.store:
#             self.store[part.get_name()] = part
#             return True
#
#         return False
#
#     def __getitem__(self, item: str) -> ParticipantScan:
#         return self.store[item]
#
#     def __contains__(self, part_name: str) -> bool:
#         return part_name in self.store
#
#     def __iter__(self) -> ParticipantScan:
#         for p in self.store:
#             yield self.store[p]
