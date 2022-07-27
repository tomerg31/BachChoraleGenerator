import torch
from abstract_chorale_objects import PhraseObject
from music21.note import Note
from music21.interval import Interval

MAX_CHORALE_LENGTH = 10
MAX_PHRASE_LENGTH_IN_QUARTERS = 20
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


def degree_to_one_hot(degree):
    one_hot_degree = torch.zeros(SCALE_DEGREES)
    one_hot_degree[degree - 1] = 1
    return one_hot_degree


def scores_to_one_hot(scores: torch.Tensor):
    one_hot = torch.zeros_like(scores)
    one_hot[torch.argmax(scores)] = 1
    return one_hot


def one_hot_to_index(one_hot):
    return int(torch.argmax(one_hot))


def degree_from_series_of_relative_degrees(series):
    output_degree = 0
    for degree in series:
        output_degree += degree
    return output_degree % SCALE_DEGREES


class PhraseTensor:
    """
    The phrase tensor object is a translation of information kept in the PhraseObject in a format that is palatable
    for torch to process. A phrase tensor will consist of the concatenation of several one-hot vectors, each one
    representing a feature in the phrase.
    """
    def __init__(self, phrase_object: PhraseObject):
        self.phrase_object = phrase_object

        # TODO: Add whether the chorale is in major or minor

        # The index of the phrase in the whole chorale
        self.index_in_chorale = torch.zeros(MAX_CHORALE_LENGTH)
        self.index_in_chorale[self.phrase_object.my_index] = 1

        # The total length of the phrase in quarters
        self.length_in_quarters = torch.zeros(MAX_PHRASE_LENGTH_IN_QUARTERS)
        assert self.phrase_object.length < MAX_PHRASE_LENGTH_IN_QUARTERS, "Phrase is longer than maximum allotted"
        self.length_in_quarters[int(self.phrase_object.length)] = 1

        # The opening and closing local tonalities. Since we are working in relative tonalities, we will not care
        # about the absolute pitch of the root. We want the harmonic function in relation to the key of the chorale
        # - meaning either TONIC, DOMINANT, or SUBDOMINANT as a classifier of the local tonality.
        self.opening_tonality = torch.zeros(len(TONAL_FUNCTIONS_TO_INDEX))
        self.opening_tonality[TONAL_FUNCTIONS_TO_INDEX[self.phrase_object.phrase_tonal_progression[OPENING][0]]] = 1

        self.closing_tonality = torch.zeros(len(TONAL_FUNCTIONS_TO_INDEX))
        self.closing_tonality[TONAL_FUNCTIONS_TO_INDEX[self.phrase_object.phrase_tonal_progression[CLOSING][0]]] = 1

        # We want four core harmonies for each phrase - pickup (if exists), opening downbeat, pre-fermata harmony,
        # and fermata harmony. In addition, we will have the soprano degree. Note that the harmony is represented by
        # the degree in relation to the *local* tonality, and the soprano degree is in relation to the

        # Pickup harmony
        self.opening_pickup_harmony = degree_to_one_hot(
            self.phrase_object.opening_harmony_group.harmony_list[0].relation_to_local_key
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.opening_pickup_harmony_soprano = degree_to_one_hot(
            self.phrase_object.opening_harmony_group.harmony_list[0].soprano_degree
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        # Downbeat harmony
        self.opening_downbeat_harmony = degree_to_one_hot(
            self.phrase_object.opening_harmony_group.harmony_list[-1].relation_to_local_key
        )

        self.opening_downbeat_harmony_soprano = degree_to_one_hot(
            self.phrase_object.opening_harmony_group.harmony_list[-1].soprano_degree
        )

        # Pre-fermata harmony
        self.pre_fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-2].relation_to_local_key
        )

        self.pre_fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-2].soprano_degree
        )

        # Fermata harmony
        self.fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-1].relation_to_local_key
        )

        self.fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-1].soprano_degree
        )

        # TODO: add inversion to harmonies

    def __call__(self):
        return torch.concat(
            [
                self.index_in_chorale,
                self.length_in_quarters,
                self.opening_tonality,
                self.closing_tonality,
                self.opening_pickup_harmony,
                self.opening_pickup_harmony_soprano,
                self.opening_downbeat_harmony,
                self.opening_downbeat_harmony_soprano,
                self.pre_fermata_harmony,
                self.pre_fermata_harmony_soprano,
                self.fermata_harmony,
                self.fermata_harmony_soprano
            ]
        )


class GeneratedPhraseTensor:
    def __init__(self, generator_output):
        self.generator_output = generator_output

        # generator_output is a 1d tensor which is exactly the shape of a PhraseTensor that came out of a real chorale.
        # The strategy to translate it back to the features that represent a chorale phrase, is to separate the long
        # tensor to its parts, and recreate a one hot encoding as in the PhraseTensor object, according to entry with
        # the highest score in each part

        cut_off_indices = torch.tensor(
            [
                MAX_CHORALE_LENGTH,
                MAX_PHRASE_LENGTH_IN_QUARTERS,
                len(TONAL_FUNCTIONS_TO_INDEX),
                len(TONAL_FUNCTIONS_TO_INDEX),
                SCALE_DEGREES,
                SCALE_DEGREES,
                SCALE_DEGREES,
                SCALE_DEGREES,
                SCALE_DEGREES,
                SCALE_DEGREES,
                SCALE_DEGREES
            ]
        )
        cut_off_indices = torch.cumsum(cut_off_indices, dim=0)

        (
            index_in_chorale_score,
            length_in_quarters_score,
            opening_tonality_score,
            closing_tonality_score,
            opening_pickup_harmony_score,
            opening_pickup_harmony_soprano_score,
            opening_downbeat_harmony_score,
            opening_downbeat_harmony_soprano_score,
            pre_fermata_harmony_score,
            pre_fermata_harmony_soprano_score,
            fermata_harmony_score,
            fermata_harmony_soprano_score
        ) = torch.tensor_split(self.generator_output, cut_off_indices)

        self.index_in_chorale_one_hot = scores_to_one_hot(index_in_chorale_score)
        self.length_in_quarters_one_hot = scores_to_one_hot(length_in_quarters_score)
        self.opening_tonality_one_hot = scores_to_one_hot(opening_tonality_score)
        self.closing_tonality_one_hot = scores_to_one_hot(closing_tonality_score)
        self.opening_pickup_harmony_one_hot = scores_to_one_hot(opening_pickup_harmony_score)
        self.opening_pickup_harmony_soprano_one_hot = scores_to_one_hot(opening_pickup_harmony_soprano_score)
        self.opening_downbeat_harmony_one_hot = scores_to_one_hot(opening_downbeat_harmony_score)
        self.opening_downbeat_harmony_soprano_one_hot = scores_to_one_hot(opening_downbeat_harmony_soprano_score)
        self.pre_fermata_harmony_one_hot = scores_to_one_hot(pre_fermata_harmony_score)
        self.pre_fermata_harmony_soprano_one_hot = scores_to_one_hot(pre_fermata_harmony_soprano_score)
        self.fermata_harmony_one_hot = scores_to_one_hot(fermata_harmony_score)
        self.fermata_harmony_soprano_one_hot = scores_to_one_hot(fermata_harmony_soprano_score)

        # Now that we have the one hot encoding for each feature, we extract the value from each feature

        self.index_in_chorale = one_hot_to_index(self.index_in_chorale_one_hot)
        self.length_in_quarters = one_hot_to_index(self.length_in_quarters_one_hot) + 1
        self.opening_tonality_degree = TONAL_FUNCTION_TO_DEGREE[
            INDEX_TO_TONAL_FUNCTIONS[
                one_hot_to_index(self.opening_tonality_one_hot)
            ]
        ]
        self.closing_tonality_degree = TONAL_FUNCTION_TO_DEGREE[
            INDEX_TO_TONAL_FUNCTIONS[
                one_hot_to_index(self.closing_tonality_one_hot)
            ]
        ]
        # Harmonic degrees are relative to their local tonality
        self.opening_pickup_harmony_degree = one_hot_to_index(self.opening_pickup_harmony_one_hot)
        self.opening_downbeat_harmony_degree = one_hot_to_index(self.opening_downbeat_harmony_one_hot)
        self.pre_fermata_harmony_degree = one_hot_to_index(self.pre_fermata_harmony_one_hot)
        self.fermata_harmony_degree = one_hot_to_index(self.fermata_harmony_one_hot)

        # Soprano degrees are relative to the harmony root
        self.opening_pickup_harmony_soprano_degree = one_hot_to_index(self.opening_pickup_harmony_soprano_one_hot)
        self.opening_downbeat_harmony_soprano_degree = one_hot_to_index(self.opening_downbeat_harmony_soprano_one_hot)
        self.pre_fermata_harmony_soprano_degree = one_hot_to_index(self.pre_fermata_harmony_soprano_one_hot)
        self.fermata_harmony_soprano_degree = one_hot_to_index(self.fermata_harmony_soprano_one_hot)

    def display_phrase_information_in_a_given_key(self, key_string='C'):
        assert key_string in KEY_TO_INT.keys(), "Key must be an uppercase letter [A-G]"
        key = KEY_TO_INT[key_string]
        print(f"*****Generated phrase information in the key of {key_string}*****")
        print(f"Index in chorale = {self.index_in_chorale}")
        print(f"Length in quarters = {self.length_in_quarters}")
        print(f"Opening tonality = {DEGREE_TO_TONAL_FUNCTION[self.opening_tonality_degree]}")
        print(f"Closing tonality = {DEGREE_TO_TONAL_FUNCTION[self.closing_tonality_degree]}")
        print("Opening pickup harmony = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.opening_tonality_degree,
                    self.opening_pickup_harmony_degree
                ])
            ]
        ))
        print("Opening pickup harmony soprano = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.opening_tonality_degree,
                    self.opening_pickup_harmony_degree,
                    self.opening_pickup_harmony_soprano_degree
                ])
            ]
        ))
        print("Opening downbeat harmony = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.opening_tonality_degree,
                    self.opening_downbeat_harmony_degree
                ])
            ]
        ))
        print("Opening downbeat harmony soprano = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.opening_tonality_degree,
                    self.opening_downbeat_harmony_degree,
                    self.opening_downbeat_harmony_soprano_degree
                ])
            ]
        ))
        print("Pre-fermata harmony = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.closing_tonality_degree,
                    self.pre_fermata_harmony_degree
                ])
            ]
        ))
        print("Pre-fermata harmony soprano = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.closing_tonality_degree,
                    self.pre_fermata_harmony_degree,
                    self.pre_fermata_harmony_soprano_degree
                ])
            ]
        ))
        print("Fermata harmony = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.closing_tonality_degree,
                    self.fermata_harmony_degree
                ])
            ]
        ))
        print("Fermata harmony soprano = {0}".format(
            INT_TO_KEY[
                degree_from_series_of_relative_degrees([
                    key,
                    self.closing_tonality_degree,
                    self.fermata_harmony_degree,
                    self.fermata_harmony_soprano_degree
                ])
            ]
        ))

# TODO: Create generator of such tensors.