import music21
import numpy as np
from music21.note import Note
from music21.interval import Interval


SOPRANO = 0
ALTO = 1
TENOR = 2
BASS = 3


def get_offset_of_note_after_fermata(chorale: music21.stream.base.Score, fermata_offset):
    note_with_fermata = chorale.parts[SOPRANO].flat.notes.getElementsByOffset(fermata_offset)[0]
    return fermata_offset + note_with_fermata.duration.quarterLength


def get_rid_of_false_measure_offsets(measure_offset_list, pickup_measure):
    assert measure_offset_list.size >= 3

    if pickup_measure:
        measure_offset_list = measure_offset_list[1:]

    # This probably isn't the best way to do this, but it is likely to assume that the second measure would be all
    # fitted in the first line, thus is a good indicator for our need
    beats_in_measure = measure_offset_list[1] - measure_offset_list[0]

    clean_measure_array = [measure_offset_list[0]]
    # Iterate over all measures and remove false measure offsets
    for i in range(1, len(measure_offset_list) - 2):
        if measure_offset_list[i + 1] - measure_offset_list[i] >= beats_in_measure or \
                measure_offset_list[i] - measure_offset_list[i - 1] >= beats_in_measure:
            clean_measure_array += [measure_offset_list[i]]

    # Add the last measure offset
    clean_measure_array += list(measure_offset_list[-2:])

    return np.array(clean_measure_array)


def get_tonal_function(baseline_harmony, harmony_root):
    """
    :param baseline_harmony: String representing a note
    :param harmony_root: String representing a note
    :return: TONIC, SUBDOMINANT or DOMINANT
    """
    # Get the directed name string of the interval between the base key and the harmony root
    interval_from_main_key = Interval(Note(baseline_harmony), Note(harmony_root)).directedName

    # Extract the numerical value of the interval
    degree = int(interval_from_main_key[-1])

    # We want to count the base key as the lower note of the interval, so in case it did not calculate it this way,
    # fix it
    if interval_from_main_key[-2] == '-':
        degree = 9 - degree

    if degree in [1, 3, 6]:
        tonal_function = 'TONIC'

    elif degree in [2, 4]:
        tonal_function = 'SUBDOMINANT'

    else:  # degree in [5, 7]
        tonal_function = 'DOMINANT'

    return tonal_function, degree


def get_degree(base: music21.note.Note, note: music21.note.Note):
    interval = Interval(base, note).directedSimpleName
    down = interval[-2] == '-'
    degree = int(interval[-1])
    if down:
        degree = (8 - degree) % 7 + 1

    return degree


class ChoraleVector:
    """
    This is the data structure we will use for learning.
    It is a representation of a bach chorale as a vector of musical phrases which are separated by fermatas.
    Each phrase is represented by a few high level features derived from the musical text, that currently include:

    1.  Opening harmony - this may include the first harmony of the phrase, or the first few harmonies if the phrase
        begins with an upbeat. This will also include a specification of the inversion, as well as the degree of the
        soprano
    2.  Closing Harmony - this may include the last harmony of the phrase, with specification of inversion and degree of
        soprano. Perhaps should also include next to last harmony in order to determine cadence.
    3.  Length of phrase (in beats? in measures?)
    4.  The position of the phrase in the chorale is represented as the position of its representation in the vector.

    """

    def __init__(self, chorale: music21.stream.base.Score):
        self.chorale = chorale

        self.chorale_name = chorale.metadata.movementName

        self.chorale_chords = chorale.chordify()

        # Most of chorales in dataset have the mode indicated. if not, we'll set the default to major.
        if 'mode' in dir(chorale.flat.keySignature):
            self.mode = chorale.flat.keySignature.mode
        else:
            self.mode = "major"

        # Since we are in Bach's music, a reliable assumption is that the last note of the bass voice would also
        # the key of the chorale
        # TODO: Should we consider major vs minor?
        self.key = chorale.parts[BASS].flat.notes.last().name

        # A boolean that indicates whether the piece begins on a pickup.
        self.pickup_measure = len(chorale.measure(0).flat.notes) != 0

        self.measure_offsets = np.array(list(chorale.measureOffsetMap().keys()))
        # A very annoying thing is that for purposes of presentation, if a bar is divided into two lines, then music21
        # counts the bar as two separate bars. This is not good for our purposes so we shall devoid the above array
        # from false barlines
        self.measure_offsets = get_rid_of_false_measure_offsets(self.measure_offsets, self.pickup_measure)

        # The meter attribute assumes that all chorales are measured in quarters
        self.meter = self.measure_offsets[1] - self.measure_offsets[0]

        self.fermata_global_offsets = np.array([n.offset for n in chorale.parts[SOPRANO].flat.notes if n.expressions])

        self.number_of_phrases = len(self.fermata_global_offsets)

        # Make the phrase representation vector
        self.phrase_object_vector = []
        for i, offset in enumerate(self.fermata_global_offsets):
            if i == 0:
                # This is the first phrase and starts in the beginning
                start_offset = 0
            else:
                # Not the first phrase, starts after most recent fermata
                start_offset = get_offset_of_note_after_fermata(chorale, self.fermata_global_offsets[i - 1])

            end_offset = offset

            self.phrase_object_vector.append(PhraseObject(self, start_offset, end_offset, i))


class PhraseObject:
    """
    The object representing each phrase in the chorale. This object will consist of key features of the phrase, such as
    the edge harmonies of the phrase.
    """
    def __init__(
            self,
            my_chorale_vector: ChoraleVector,
            start_offset,
            end_offset,
            index_in_vector
    ):
        self.my_chorale_vector = my_chorale_vector

        self.my_index = index_in_vector

        self.start_offset = start_offset

        self.end_offset = end_offset

        self.pickup = start_offset not in self.my_chorale_vector.measure_offsets

        self.first_downbeat = start_offset if not self.pickup else \
            self.my_chorale_vector.measure_offsets[self.my_chorale_vector.measure_offsets > start_offset].min()

        # It is possible that the first downbeat is a rest. If so, redefine first downbeat as the downbeat of the
        # following measure
        if not self.my_chorale_vector.chorale_chords.recurse().getElementsByOffsetInHierarchy(self.first_downbeat). \
                getElementsByClass(music21.note.NotRest):
            self.first_downbeat = self.first_downbeat + self.my_chorale_vector.meter

        self.opening_harmony_group = self.get_harmony_group(start_offset, opening=True)

        self.closing_harmony_group = self.get_harmony_group(end_offset, opening=False)

        self.phrase_tonal_progression = self.get_phrase_tonal_progression()

        self.length = end_offset - start_offset

        for harmony in self.opening_harmony_group.harmony_list:
            harmony.set_relation_to_local_key()

        for harmony in self.closing_harmony_group.harmony_list:
            harmony.set_relation_to_local_key()

    def get_harmony_group(self, offset, opening: bool):
        # This method will return a HarmonyGroup object
        # For the opening group, we want all harmonies from the pick-up to the downbeat (usually this will be 2
        # harmonies, and if there is no pickup, just the one).
        # For the closing group, we want the two last harmonies - the one of the fermata, and the one preceding it,
        # so we can determine the type of cadence.
        if opening:
            return HarmonyGroup(self.my_chorale_vector.chorale_chords.recurse().
                                getElementsByOffsetInHierarchy(offset, self.first_downbeat, includeEndBoundary=True).
                                getElementsByClass('Chord'),
                                self,
                                opening)
        else:
            return HarmonyGroup(self.my_chorale_vector.chorale_chords.recurse().
                                getElementsByOffsetInHierarchy(offset - 2, offset, includeEndBoundary=True).
                                getElementsByClass('Chord'),
                                self,
                                opening)

    def get_phrase_tonal_progression(self):
        # There are three general tonal functions: TONIC, SUBDOMINANT, and DOMINANT. Those can be extended to represent
        # each of the scale degrees as follows:
        # TONIC: I, III, VI
        # SUBDOMINANT: II, IV
        # DOMINANT: V, VII
        # We shall return a pair of tonal functions representing the local tonality of the opening harmony group and
        # the closing harmony group
        return (self.opening_harmony_group.local_tonality,
                self.closing_harmony_group.local_tonality)


class HarmonyGroup:
    def __init__(self, chord_iterator, my_phrase, opening=True):
        self.it = chord_iterator

        self.my_phrase = my_phrase

        self.chorale_key = my_phrase.my_chorale_vector.key

        self.harmony_list = [MyHarmony(my_chord, self) for my_chord in chord_iterator]

        self.opening = opening

        self.final_cadence = False if my_phrase is None else \
            my_phrase.my_index == my_phrase.my_chorale_vector.number_of_phrases - 1

        self.cadence_type = None if opening else self.get_cadence_type()

        # The local tonality will be represented by a tuple consisting of the tonal functionality of the harmony in
        # relation to the main key, and the degree of the main harmony in relation to the main chorale key.
        self.local_tonality = self.get_local_tonality()

    def get_cadence_type(self):
        assert len(self.harmony_list) > 1

        # AUTHENTIC CADENCE: V-I transition. If soprano ends on the tonic we'll call it PERFECT.
        # We want two things - 1. Bass either ascends by a fourth or descends by a fifth. (Maybe there should
        # be a distinction in score - whatever score means right now - as the latter is stronger), and 2. The leading
        # harmony must be either a major chord or a dominant seventh.

        # If the second condition above is not satisfied then the cadence would be deemed as HALF CADENCE.
        # If the motion in the bass is of an ascending or descending second, this would be a HALF CADENCE.
        # If bass ascending by a major second, then first harmony mustn't be a dominant seventh.
        # Harmony on fermata MUST be major chord.
        # When cadence type is detected to be a half cadence, then we shall deem the last harmony of the measure as
        # the fifth degree, and the local key as whatever of which said harmony is the fifth degree.

        # PLAGAL CADENCE: IV-I transition. Bass ascending by fifth or descending by fourth.
        # How do we distinguish between plagal and half? difficult. Such harmonic transition at the very end of the
        # chorale will be plagal. Let's decide for now that plagal cadence not in the last phrase is HALF CADENCE,
        # unless harmony in fermata is minor.

        # DECEPTIVE CADENCE: We will limit this to V-VI transition. To distinguish between deceptive and half we shall
        # decide that if second harmony is minor it is then a deceptive cadence, or if the first harmony is a dominant
        # seventh.

        # We will take as the first harmony the harmony that is on the quarter, which is not the same harmony as the
        # last harmony.
        second_harmony = self.harmony_list[-1].chord
        second_bass = second_harmony.bass()

        # Have a default value for first harmony
        first_harmony = self.harmony_list[-2].chord
        found = False
        i = len(self.harmony_list) - 2
        while not found:
            if i == 0 or \
                    (self.harmony_list[i].chord.isMajorTriad() and
                     Interval(self.harmony_list[i].chord.bass, second_bass).simpleName != 'P1') or \
                    self.harmony_list[i].chord.isDominantSeventh():
                first_harmony = self.harmony_list[i].chord
                found = True
                self.harmony_list = self.harmony_list[i:]
            i -= 1

        first_bass = first_harmony.root()

        bass_interval = Interval(first_bass, second_bass).directedName

        second_soprano = second_harmony.notes[-1].pitch

        if (first_harmony.isMajorTriad() or first_harmony.isDominantSeventh()) and \
                (bass_interval == 'P4' or bass_interval == 'P-5'):
            return 'AUTHENTIC' if second_bass.name != second_soprano.name else 'PERFECT'

        if (bass_interval == 'P-4' or bass_interval == 'P5') and self.final_cadence:
            return 'PLAGAL'

        if (first_harmony.isMajorTriad() or first_harmony.isDominantSeventh()) and \
                (bass_interval == 'M2' or bass_interval == 'm2'):
            return 'DECEPTIVE'

        return 'HALF'

    def get_local_tonality(self):
        """
        :return: Tuple of the name of the harmonic function of the local tonality, and the degree of the local
        harmony in relation to the chorale key
        """
        if self.opening:
            # This is the opening harmonies and we will determine the tonality according to the harmony on the downbeat
            harmony_root = self.harmony_list[-1].chord.root().name

        else:
            # The tonality in the closing group will be dependent on the type of cadence
            assert self.cadence_type is not None

            if self.cadence_type == 'AUTHENTIC' or self.cadence_type == 'PERFECT' or self.cadence_type == 'PLAGAL':
                harmony_root = self.harmony_list[-1].chord.root().name

            else:
                # self.cadence_type == 'HALF':
                harmony_root = self.harmony_list[-1].chord.root().transpose(Interval('P4')).name

        return get_tonal_function(self.chorale_key, harmony_root)


class MyHarmony:
    def __init__(self, chord, my_harmony_group: HarmonyGroup):
        self.chord = chord
        self.my_harmony_group = my_harmony_group
        self.metric_weight = None

        self.root = Note(self.chord.root())

        self.relation_to_chorale_key = self.get_relation_to_chorale_key()

        # The phrase key is determined by the harmonies, meaning after this object is created. This should be
        # filled at a later time after the generation of the chorale representation.
        self.relation_to_local_key = None

        self.soprano_degree = self.get_soprano_degree()
        self.inversion = chord.inversion()

    def get_relation_to_chorale_key(self):
        # For now this will return the degree of the harmony in the context of the main key
        main_key = Note(self.my_harmony_group.my_phrase.my_chorale_vector.key)
        return get_degree(main_key, self.root)

    def set_relation_to_local_key(self):
        # Local key degree is the local tonality in relation to the chorale key
        local_key_degree = self.my_harmony_group.local_tonality[1]

        degree = (self.relation_to_chorale_key - local_key_degree) % 7 + 1

        self.relation_to_local_key = degree

    def set_metric_weight(self):
        raise NotImplementedError

    def get_soprano_degree(self):
        soprano_note = self.chord.notes[-1]
        return get_degree(self.root, soprano_note)

# TODO: Think about whether next step is to learn those representations and have an automatic generator generate these,
#  ORRRR to understand how to transcribe those representations back to skeletons of chorales to set as inputs for
#  DEEPBACH....???


# TODO: What if the harmony on the downbeat after a fermata is a harmonic appogiattura?
