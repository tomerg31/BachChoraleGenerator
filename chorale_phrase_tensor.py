import torch
from abstract_chorale_objects import PhraseObject
from chorale_feature_extractor import ExtractedPhrase

MAX_CHORALE_LENGTH = 24
MAX_PHRASE_LENGTH_IN_QUARTERS = 24
NUMBER_OF_MODES = 2
TONAL_FUNCTIONS_TO_INDEX = {
    'TONIC': 0,
    'SUBDOMINANT': 1,
    'DOMINANT': 2
}
INDEX_TO_TONAL_FUNCTIONS = {
    0: 'TONIC',
    1: 'SUBDOMINANT',
    2: 'DOMINANT'
}
TONAL_FUNCTION_TO_DEGREE = {
    'TONIC': 0,
    'SUBDOMINANT': 3,
    'DOMINANT': 4
}
DEGREE_TO_TONAL_FUNCTION = {
    0: 'TONIC',
    3: 'SUBDOMINANT',
    4: 'DOMINANT'
}

OPENING = 0
CLOSING = 1

SCALE_DEGREES = 7
KEY_TO_INT = {
    'C': 0,
    'D': 1,
    'E': 2,
    'F': 3,
    'G': 4,
    'A': 5,
    'B': 6
}
INT_TO_KEY = {
    0: 'C',
    1: 'D',
    2: 'E',
    3: 'F',
    4: 'G',
    5: 'A',
    6: 'B'
}
INT_TO_MODE = {
    0: 'MAJOR',
    1: 'MINOR'
}
# 0 will be the root inversion, 1 - first inversion, 2 - second inversion, and 3 - "other"
NUMBER_OF_INVERSIONS = 4

FEATURES_ONE_HOT_LENGTH = {
    'index_in_chorale': 10,
    'phrase_mode': 2,
    'length_in_quarters': 24,
    'opening_pickup_harmony': 7,
    'opening_pickup_harmony_soprano': 7,
    'opening_pickup_harmony_inversion': 4,
    'opening_downbeat_harmony': 7,
    'opening_downbeat_harmony_soprano': 7,
    'opening_downbeat_harmony_inversion': 4,
    'pre_fermata_harmony': 7,
    'pre_fermata_harmony_soprano': 7,
    'pre_fermata_harmony_inversion': 4,
    'fermata_harmony': 7,
    'fermata_harmony_soprano': 7,
    'fermata_harmony_inversion': 4
}

# This is the list of features that the phrase vector that is the input and the output of the VAE will have.
# If you wish to add another feature, make sure it is supported in all the structures in this file.
PHRASE_FEATURE_LIST = [
    # 'index_in_chorale'
    # 'phrase_mode',
    # 'length_in_quarters',
    'opening_pickup_harmony',
    # 'opening_pickup_harmony_soprano',
    # 'opening_pickup_harmony_inversion',
    'opening_downbeat_harmony',
    # 'opening_downbeat_harmony_soprano',
    # 'opening_downbeat_harmony_inversion',
    'pre_fermata_harmony',
    # 'pre_fermata_harmony_soprano',
    # 'pre_fermata_harmony_inversion',
    'fermata_harmony',
    # 'fermata_harmony_soprano',
    # 'fermata_harmony_inversion'
]


def degree_to_one_hot(degree):
    one_hot_degree = torch.zeros(SCALE_DEGREES)
    one_hot_degree[degree - 1] = 1
    return one_hot_degree


def scores_to_index(scores: torch.Tensor):
    return torch.argmax(scores, dim=-1)


def degree_from_series_of_relative_degrees(series):
    output_degree = 0
    for degree in series:
        output_degree += degree
    return output_degree % SCALE_DEGREES


def inversion_to_one_hot(inversion):
    one_hot = torch.zeros(NUMBER_OF_INVERSIONS)
    if inversion >= 3:
        one_hot[3] = 1
    else:
        one_hot[inversion] = 1
    return one_hot


def cut_tensor_by_features(tensor):
    cut_off_indices = torch.tensor(
        [FEATURES_ONE_HOT_LENGTH[feature] for feature in PHRASE_FEATURE_LIST][:-1]
    )
    cut_off_indices = torch.cumsum(cut_off_indices, dim=0)

    return torch.tensor_split(tensor, cut_off_indices, dim=-1)


class PhraseTensor:
    """
    The phrase tensor object is a translation of information kept in the PhraseObject in a format that is palatable
    for torch to process. A phrase tensor will consist of the concatenation of several one-hot vectors, each one
    representing a feature in the phrase.
    """

    def __init__(self, phrase_object: ExtractedPhrase):
        self.phrase_object = phrase_object
        self.valid = True

        # The index of the phrase in the whole chorale
        self.index_in_chorale = torch.zeros(MAX_CHORALE_LENGTH)
        self.index_in_chorale[self.phrase_object.index_in_chorale] = 1

        self.phrase_mode = torch.zeros(NUMBER_OF_MODES)
        mode_index = 0 if self.phrase_object.chorale_mode != 'minor' else 1
        self.phrase_mode[mode_index] = 1

        # The total length of the phrase in quarters
        self.length_in_quarters = torch.zeros(MAX_PHRASE_LENGTH_IN_QUARTERS)

        # Too long phrases are outliers and should probably not be dealt with.
        if self.phrase_object.phrase_length >= MAX_PHRASE_LENGTH_IN_QUARTERS:
            print(f"Phrase is longer than maximum allotted. Chorale "
                  f"{self.phrase_object.chorale_name} "
                  f"phrase index {self.phrase_object.index_in_chorale}")
            # Mark this phrase as too long and remove it when parsing that data
            self.valid = False
            return

        self.length_in_quarters[int(self.phrase_object.phrase_length)] = 1

        # We want four core harmonies for each phrase - pickup (if exists), opening downbeat, pre-fermata harmony,
        # and fermata harmony. In addition, we will have the soprano degree. Note that the harmony is represented by
        # the degree in relation to the *local* tonality, and the soprano degree is in relation to the

        # UPDATE: Now we attempt at disregarding the local harmony, and just look at the degrees of the
        # harmonies in relation to the main key of the chorale

        # Pickup harmony
        self.opening_pickup_harmony = degree_to_one_hot(
            self.phrase_object.opening_harmony_group[0].relation_to_chorale_key
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.opening_pickup_harmony_soprano = degree_to_one_hot(
            self.phrase_object.opening_harmony_group[0].soprano_degree
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.opening_pickup_harmony_inversion = inversion_to_one_hot(
            self.phrase_object.opening_harmony_group[0].inversion
        ) if self.phrase_object.pickup else torch.zeros(NUMBER_OF_INVERSIONS)

        # Downbeat harmony
        self.opening_downbeat_harmony = degree_to_one_hot(
            self.phrase_object.opening_harmony_group[-1].relation_to_chorale_key
        )

        self.opening_downbeat_harmony_soprano = degree_to_one_hot(
            self.phrase_object.opening_harmony_group[-1].soprano_degree
        )

        self.opening_downbeat_harmony_inversion = inversion_to_one_hot(
            self.phrase_object.opening_harmony_group[-1].inversion
        )

        # Pre-fermata harmony
        self.pre_fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group[-2].relation_to_chorale_key
        )

        self.pre_fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group[-2].soprano_degree
        )

        self.pre_fermata_harmony_inversion = inversion_to_one_hot(
            self.phrase_object.closing_harmony_group[-2].inversion
        )

        # Fermata harmony
        self.fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group[-1].relation_to_chorale_key
        )

        self.fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group[-1].soprano_degree
        )

        self.fermata_harmony_inversion = inversion_to_one_hot(
            self.phrase_object.closing_harmony_group[-1].inversion
        )

    def __call__(self):
        features = []
        for feature in PHRASE_FEATURE_LIST:
            exec(f"features += [self.{feature}]")

        return torch.concat(features)


class GeneratedPhraseBatchTensor:
    def __init__(self, generator_output: torch.Tensor):
        self.generator_output = generator_output
        self.batch_size = self.generator_output.size()[0]
        self.phrase_vector_length = self.generator_output.size()[-1]

        # We want to support both a single phrase and a batch of phrases, so make sure that if the input is a
        # single phrase, we treat it as a batch of size one.
        if len(self.generator_output.size()) == 1:
            self.generator_output = torch.unsqueeze(self.generator_output, dim=0)
            self.batch_size = 1

        # The strategy to translate it back to the features that represent a chorale phrase, is to separate the long
        # tensor to its parts, and recreate a one hot encoding as in the PhraseTensor object, according to entry with
        # the highest score in each part

        # This list is ordered like PHRASE_FEATURE_LIST at the top of this file
        self.phrase_feature_list = cut_tensor_by_features(self.generator_output)

        # Initialize all the potential features of the phrase
        self.index_in_chorale = None
        self.phrase_mode = None
        self.length_in_quarters = None
        self.opening_pickup_harmony = None
        self.opening_pickup_harmony_soprano = None
        self.opening_pickup_harmony_inversion = None
        self.opening_downbeat_harmony = None
        self.opening_downbeat_harmony_soprano = None
        self.opening_downbeat_harmony_inversion = None
        self.pre_fermata_harmony = None
        self.pre_fermata_harmony_soprano = None
        self.pre_fermata_harmony_inversion = None
        self.fermata_harmony = None
        self.fermata_harmony_soprano = None
        self.fermata_harmony_inversion = None

        # Set the features to their actual value according to the generator output
        for i, feature in enumerate(PHRASE_FEATURE_LIST):
            exec(f'self.{feature} = scores_to_index(self.phrase_feature_list[i])')

    def display_phrase_information_in_a_given_key(self, key_string=None, display_index_range=None):
        assert (key_string in KEY_TO_INT.keys()) or (key_string is None), "Key must be an uppercase letter [A-G]"

        if not display_index_range:
            # Display all phrases in tensor
            display_index_range = range(self.batch_size)
        if isinstance(display_index_range, int):
            assert display_index_range < self.batch_size, f"Only {self.batch_size} phrases in tensor!"
            display_index_range = range(display_index_range, display_index_range + 1)

        for i in display_index_range:
            mode = ''
            if self.phrase_mode is not None:
                mode = INT_TO_MODE[self.phrase_mode[i].item()]

            if not key_string:
                if mode == 'MAJOR' or mode == '':
                    key_string = 'C'
                else:
                    key_string = 'A'

            key = KEY_TO_INT[key_string]

            print(f"***** Generated phrase {i} information in the key of {key_string} {mode} *****")

            if self.index_in_chorale is not None:
                print(f"Index in chorale = {self.index_in_chorale[i]}")

            if self.length_in_quarters is not None:
                # Note that this is calculated according to the index, thus we need to add one to get the actual length
                print(f"Length in quarters = {self.length_in_quarters[i] + 1}")

            if self.opening_pickup_harmony is not None:
                print("Opening pickup harmony = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.opening_pickup_harmony[i].item()
                        ])
                    ]
                ))

            if self.opening_pickup_harmony_soprano is not None:
                print("Opening pickup harmony soprano = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.opening_pickup_harmony[i].item(),
                            self.opening_pickup_harmony_soprano[i].item()
                        ])
                    ]
                ))

            if self.opening_pickup_harmony_inversion is not None:
                print("Opening pickup harmony inversion = {0}".format(
                    self.opening_pickup_harmony_inversion[i].item()
                ))

            if self.opening_downbeat_harmony is not None:
                print("Opening downbeat harmony = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.opening_downbeat_harmony[i].item()
                        ])
                    ]
                ))

            if self.opening_downbeat_harmony_soprano is not None:
                print("Opening downbeat harmony soprano = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.opening_downbeat_harmony[i].item(),
                            self.opening_downbeat_harmony_soprano[i].item()
                        ])
                    ]
                ))

            if self.opening_downbeat_harmony_inversion is not None:
                print("Opening downbeat harmony inversion = {0}".format(
                    self.opening_downbeat_harmony_inversion[i].item()
                ))

            if self.pre_fermata_harmony is not None:
                print("Pre-fermata harmony = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.pre_fermata_harmony[i].item()
                        ])
                    ]
                ))

            if self.pre_fermata_harmony_soprano is not None:
                print("Pre-fermata harmony soprano = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.pre_fermata_harmony[i].item(),
                            self.pre_fermata_harmony_soprano[i].item()
                        ])
                    ]
                ))

            if self.pre_fermata_harmony_inversion is not None:
                print("Pre-fermata harmony inversion = {0}".format(
                    self.pre_fermata_harmony_inversion[i].item()
                ))

            if self.fermata_harmony is not None:
                print("Fermata harmony = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.fermata_harmony[i].item()
                        ])
                    ]
                ))

            if self.fermata_harmony_soprano is not None:
                print("Fermata harmony soprano = {0}".format(
                    INT_TO_KEY[
                        degree_from_series_of_relative_degrees([
                            key,
                            self.fermata_harmony[i].item(),
                            self.fermata_harmony_soprano[i].item()
                        ])
                    ]
                ))

            if self.fermata_harmony_inversion is not None:
                print("Fermata harmony inversion = {0}".format(
                    self.fermata_harmony_inversion[i].item()
                ))
