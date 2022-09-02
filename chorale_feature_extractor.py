import music21
import numpy as np
from music21.note import Note
from music21.interval import Interval

SOPRANO = 0
ALTO = 1
TENOR = 2
BASS = 3


def get_degree(base: music21.note.Note, note: music21.note.Note):
    interval = Interval(base, note).directedSimpleName
    down = interval[-2] == '-'
    degree = int(interval[-1])
    if down:
        degree = (8 - degree) % 7 + 1

    return degree


class ExtractedChorale:
    def __init__(self, chorale: music21.stream.base.Score):
        self.valid = True

        self.chorale_name = chorale.metadata.movementName

        # If the chorale does not have the name, then it most likely means that the data is not so good, so we can skip
        # it
        if not self.chorale_name:
            self.valid = False
            return

        self.key, self.mode = self._get_key_and_mode_of_chorale(chorale)

        # A boolean that indicates whether the piece begins on a pickup.
        self.pickup_measure = len(chorale.measure(0).flat.notes) != 0

        self.measure_offsets = np.array(list(chorale.measureOffsetMap().keys()))

        # The meter attribute assumes that all chorales are measured in quarters
        self.meter_in_quarters = self._get_meter_in_quarters(chorale)
        if self.meter_in_quarters == -1:
            self.valid = False
            return

        # A very annoying thing is that for purposes of presentation, if a bar is divided into two lines, then music21
        # counts the bar as two separate bars. This is not good for our purposes so we shall devoid the above array
        # from false bar-lines
        self.measure_offsets = self._remove_false_measure_offsets()

        self.fermata_global_offsets = np.array(
            [
                note.offset for note in chorale.parts[SOPRANO].flat.notes if True in
                [isinstance(expression, music21.expressions.Fermata) for expression in note.expressions]
            ]
        )

        self.number_of_phrases = len(self.fermata_global_offsets)

        # The most important field in the class
        self.phrase_vector = []
        for i, offset in enumerate(self.fermata_global_offsets):
            if i == 0:
                # This is the first phrase and starts in the beginning
                start_offset = 0
            else:
                # Not the first phrase, starts after most recent fermata
                start_offset = self._get_offset_of_note_after_fermata(chorale, phrase_index=(i - 1))

            end_offset = offset
            start_of_next_phrase_offset = self._get_offset_of_note_after_fermata(chorale, phrase_index=i)
            self.phrase_vector.append(
                ExtractedPhrase(
                    chorale.chordify(),
                    self.chorale_name,
                    i,
                    start_offset,
                    end_offset,
                    start_of_next_phrase_offset,
                    self.key,
                    self.mode,
                    [measure_offset for measure_offset in self.measure_offsets if
                     start_offset <= measure_offset <= end_offset],
                    len(self.fermata_global_offsets)
                )
            )

    @staticmethod
    def _get_key_and_mode_of_chorale(chorale):
        if 'tonic' in dir(chorale.flat.keySignature):
            key = chorale.flat.keySignature.tonic.name
        else:
            # Since we are in Bach's music, a reliable assumption is that the last note of the bass voice would also
            # the key of the chorale, so if the information is not already provided, we can use that
            key = chorale.parts[BASS].flat.notes.last().name

        # Most of chorales in the dataset have the mode indicated. if not, we'll set the default to major.
        if 'mode' in dir(chorale.flat.keySignature):
            mode = chorale.flat.keySignature.mode
        else:
            mode = "major"

        return key, mode

    def _get_meter_in_quarters(self, chorale):
        # It is possible that some chorales have a meter change in the middle. For now we will regard that as an
        # untreated edge case

        time_signature = chorale.flat.timeSignature
        print(self.chorale_name)
        if 'numerator' in dir(time_signature) and 'denominator' in dir(time_signature):
            # We assume we are not handling fancy time signatures
            if not (time_signature.denominator == 2 or time_signature.denominator == 4):
                return -1
            meter = time_signature.numerator * 4 / time_signature.denominator

        else:
            # If information not available in input chorale, we will extract it from the measure offsets. Note that
            # we still need to fix the measure offsets at this point, but we assume that the first two measures are
            # correctly placed
            meter = self.measure_offsets[1] - self.measure_offsets[0]

        return meter

    def _remove_false_measure_offsets(self):
        assert self.measure_offsets.size >= 3

        # If the chorale has a pickup, do not count the pickup as a whole measure
        if self.pickup_measure:
            self.measure_offsets = self.measure_offsets[1:]

        clean_measure_array = [self.measure_offsets[0]]
        # Iterate over all measures and remove false measure offsets
        for i in range(1, len(self.measure_offsets) - 2):
            if (
                (self.measure_offsets[i + 1] - self.measure_offsets[i] >= self.meter_in_quarters) or
                (self.measure_offsets[i] - self.measure_offsets[i - 1] >= self.meter_in_quarters)
            ):
                clean_measure_array += [self.measure_offsets[i]]

        # Add the last measure offset
        clean_measure_array += list(self.measure_offsets[-2:])

        return np.array(clean_measure_array)

    def _get_offset_of_note_after_fermata(self, chorale, phrase_index):
        previous_fermata_offset = self.fermata_global_offsets[phrase_index]
        note_with_fermata = chorale.parts[SOPRANO].flat.notes.getElementsByOffset(previous_fermata_offset)[0]
        return previous_fermata_offset + note_with_fermata.duration.quarterLength


class ExtractedPhrase:
    """
    The object representing each phrase in the chorale. This object will consist of key features of the phrase, such as
    the edge harmonies of the phrase.
    """

    def __init__(
            self,
            chorale_chords,
            chorale_name,
            index_in_chorale,
            start_offset,
            end_offset,
            start_of_next_phrase_offset,
            chorale_key,
            chorale_mode,
            measure_offsets,
            number_of_phrases_in_chorale
    ):
        self.chorale_name = chorale_name

        self.index_in_chorale = index_in_chorale

        self.start_offset = start_offset

        self.end_offset = end_offset

        self.chorale_key = chorale_key

        self.chorale_mode = chorale_mode

        self.measure_offsets = measure_offsets

        self.number_of_phrases_in_chorale = number_of_phrases_in_chorale

        self.final_phrase_in_chorale = self.index_in_chorale == self.number_of_phrases_in_chorale - 1

        self.pickup = start_offset not in self.measure_offsets

        self.phrase_length = start_of_next_phrase_offset - start_offset

        self.first_downbeat_offset = self._get_first_downbeat_offset(chorale_chords)

        self.opening_harmony_group = self._get_harmony_group(chorale_chords, opening=True)

        self.closing_harmony_group = self._get_harmony_group(chorale_chords, opening=False)

        self.cadence_type = self._get_cadence_type()

        self.opening_tonality = self._get_tonality(opening=True)

        self.closing_tonality = self._get_tonality(opening=False)

    def _get_first_downbeat_offset(self, chorale_chords):
        # This is in case the first beat of the phrase is on the downbeat, but it is a rest
        if not (
                chorale_chords.recurse().getElementsByOffsetInHierarchy(self.measure_offsets[0]).
                getElementsByClass(music21.note.NotRest)
        ):
            self.pickup = True
            return self.measure_offsets[1]

        return self.measure_offsets[0]

    def _get_harmony_group(self, chorale_chords, opening):
        if opening:
            group_start_offset = self.start_offset
            group_end_offset = self.first_downbeat_offset
        else:
            group_start_offset = self.end_offset - 2
            group_end_offset = self.end_offset

        chord_iterator = (
            chorale_chords.recurse().
            getElementsByOffsetInHierarchy(group_start_offset, group_end_offset, includeEndBoundary=True).
            getElementsByClass('Chord')
        )
        return [MyHarmony(my_chord, self.chorale_key) for my_chord in chord_iterator]

    def _get_cadence_type(self):
        assert len(self.closing_harmony_group) > 1

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
        second_harmony = self.closing_harmony_group[-1].chord
        second_bass = second_harmony.bass()

        # Have a default value for first harmony
        first_harmony = self.closing_harmony_group[-2].chord
        found = False
        i = len(self.closing_harmony_group) - 2
        while not found:
            if i == 0 or \
                    (self.closing_harmony_group[i].chord.isMajorTriad() and
                     Interval(self.closing_harmony_group[i].chord.bass, second_bass).simpleName != 'P1') or \
                    self.closing_harmony_group[i].chord.isDominantSeventh():
                first_harmony = self.closing_harmony_group[i].chord
                found = True
                self.closing_harmony_group = self.closing_harmony_group[i:]
            i -= 1

        first_bass = first_harmony.root()

        bass_interval = Interval(first_bass, second_bass).directedName

        second_soprano = second_harmony.notes[-1].pitch

        if (first_harmony.isMajorTriad() or first_harmony.isDominantSeventh()) and \
                (bass_interval == 'P4' or bass_interval == 'P-5'):
            return 'AUTHENTIC' if second_bass.name != second_soprano.name else 'PERFECT'

        if (bass_interval == 'P-4' or bass_interval == 'P5') and self.final_phrase_in_chorale:
            return 'PLAGAL'

        if (first_harmony.isMajorTriad() or first_harmony.isDominantSeventh()) and \
                (bass_interval == 'M2' or bass_interval == 'm2'):
            return 'DECEPTIVE'

        return 'HALF'

    def _get_tonality(self, opening):
        """
        :return: The int value of the degree of the tonality in relation to the main key of the chorale
        """
        if opening:
            # This is the opening harmonies and we will determine the tonality according to the harmony on the downbeat

            print(self.chorale_name, self.index_in_chorale)
            return self.opening_harmony_group[-1].relation_to_chorale_key

        else:
            # The tonality in the closing group will be dependent on the type of cadence
            assert self.cadence_type is not None

            if self.cadence_type == 'AUTHENTIC' or self.cadence_type == 'PERFECT' or self.cadence_type == 'PLAGAL':
                return self.closing_harmony_group[-1].relation_to_chorale_key

            else:
                # self.cadence_type == 'HALF':
                return (self.closing_harmony_group[-1].relation_to_chorale_key + 3) % 7


class MyHarmony:
    def __init__(self, chord, chorale_key):
        self.chord = chord

        self.chorale_key = chorale_key

        self.metric_weight = None

        self.root = Note(chord.root())

        self.relation_to_chorale_key = self._get_relation_to_chorale_key()

        self.soprano_degree = self._get_soprano_degree(chord)

        self.inversion = chord.inversion()

    def _get_relation_to_chorale_key(self):
        # For now this will return the degree of the harmony in the context of the main key
        main_key = Note(self.chorale_key)
        return get_degree(main_key, self.root)

    def _set_metric_weight(self):
        raise NotImplementedError

    def _get_soprano_degree(self, chord):
        soprano_note = chord.notes[-1]
        return get_degree(self.root, soprano_note)


class ExtractedChoralesContainer:
    def __init__(self, iterator=None):
        self.chorale_dict = {}
        if iterator:
            self.add_chorales_from_music21_iterator(iterator)

    def add_chorales_from_music21_iterator(self, iterator):
        for chorale in iterator:
            new_chorale_vector = ExtractedChorale(chorale)
            if new_chorale_vector.valid:
                self.chorale_dict[new_chorale_vector.chorale_name] = new_chorale_vector

    def add_extracted_chorale(self, extracted_chorale: ExtractedChorale):
        self.chorale_dict[extracted_chorale.chorale_name] = extracted_chorale

    def __getitem__(self, item):
        return self.chorale_dict[item]



