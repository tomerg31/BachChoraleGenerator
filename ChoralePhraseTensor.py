import torch
from abstract_chorale_objects import PhraseObject

MAX_CHORALE_LENGTH = 10
MAX_PHRASE_LENGTH_IN_QUARTERS = 20
TONAL_FUNCTIONS = {
    'TONIC': 0,
    'SUBDOMINANT': 1,
    'DOMINANT': 2
}
OPENING = 0
CLOSING = 1

SCALE_DEGREES = 7


def degree_to_one_hot(degree):
    one_hot_degree = torch.zeros(SCALE_DEGREES)
    one_hot_degree[degree - 1] = 1
    return one_hot_degree


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
        self.opening_tonality = torch.zeros(len(TONAL_FUNCTIONS))
        self.opening_tonality[TONAL_FUNCTIONS[self.phrase_object.phrase_tonal_progression[OPENING][0]]] = 1

        self.closing_tonality = torch.zeros(len(TONAL_FUNCTIONS))
        self.closing_tonality[TONAL_FUNCTIONS[self.phrase_object.phrase_tonal_progression[CLOSING][0]]] = 1

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
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.opening_downbeat_harmony_soprano = degree_to_one_hot(
            self.phrase_object.opening_harmony_group.harmony_list[-1].soprano_degree
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        # Pre-fermata harmony
        self.pre_fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-2].relation_to_local_key
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.pre_fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-2].soprano_degree
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        # Fermata harmony
        self.fermata_harmony = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-1].relation_to_local_key
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

        self.fermata_harmony_soprano = degree_to_one_hot(
            self.phrase_object.closing_harmony_group.harmony_list[-1].soprano_degree
        ) if self.phrase_object.pickup else torch.zeros(SCALE_DEGREES)

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
