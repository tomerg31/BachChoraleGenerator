import music21
import torch

# This is an attempt to translate a phrase from a chorale to a piano roll image, then perhaps use some convolutional
# network to help with the learning

# TODO: find whet this is and fill (in semitones)
LARGEST_RANGE_IN_CHORALES = 42

# When we find out what we want the range to be, we will set that the lowest possible note will be in index 0, and
# probably we'll want two indexing for the height of notes starting from the lowest note of the chorale, one for the
# diatonic distances, and one for chromatic distances.
highest_note_index = 42


def phrase_to_image(chorale_chords, diatonic=False) -> torch.Tensor:
    """
    This function will be called from the chorale feature extractor and so we use this interface for convenience.
    The generated "image" from the chorale phrase will be a two dimensional tensor s.t. one dimension represents
    time, quantized to sixteenth notes, meaning that it will be the length of 4 * (end_offset - start_offset) as
    the offsets are measured in quarter notes. The second dimension (height) is the location of the notes. I think
    it is worth quantizing it in both full chromatic resolution and in a reduced diatonic one.
    :param chorale_chords: music21 chord stream
    :param diatonic:
    :return: Phrase image - torch.Tensor
    """

    # Set the range of the image. To do this we want to check what is the biggest distance between the highest note in
    # the soprano voice and the lowest note in the bass voice in all the chorales
    height = int(LARGEST_RANGE_IN_CHORALES * 12 / 7) if diatonic else LARGEST_RANGE_IN_CHORALES

    width = 4 * (end_offset - start_offset)

    # Initialize an empty tensor with the appropriate dimensions
    image = torch.zeros((height, width))




