from chorale_phrase_tensor import PhraseTensor, PHRASE_FEATURE_LIST
from chorale_feature_extractor import ExtractedPhrase, ExtractedChoralesContainer
import pickle
import os
from typing import Union
import torch

# The idea here is to create a vocabulary where each word is a possible phrase. We want to extract features in a word
# such that on one hand we will have valuable defining features, and on the other hand, that these features will be
# general enough so that not each phrase will be unique. We will try various combinations of the features extracted
# by the PhraseTensor object. Given those, our vocabulary will be comprised of all existing unique feature combinations
# and each phrase will get an index according to its place in the vocabulary.


class BagOfPhrases:
    def __init__(
            self,
            extracted_chorale_container: Union[ExtractedChoralesContainer, str],
            feature_list=PHRASE_FEATURE_LIST):
        # If extracted_chorale_container is a path to a pickle file where the picked container is saved, load it
        if isinstance(extracted_chorale_container, str) and os.path.exists(extracted_chorale_container):
            assert extracted_chorale_container[-3:] == 'pkl'
            pickle_file = open(extracted_chorale_container, 'rb')
            extracted_chorale_container = pickle.load(pickle_file)
            pickle_file.close()

        assert isinstance(extracted_chorale_container, ExtractedChoralesContainer)

        # Go over all phrases in all chorales in container and turn them into PhraseItemInBag object. Each
        self.index_to_phrase_items_map: dict[int, Union[list[PhraseItemInBag], str]] = {}

        for chorale_name in extracted_chorale_container.chorale_dict.keys():
            for phrase in extracted_chorale_container.chorale_dict[chorale_name].phrase_vector:
                new_phrase_item = PhraseItemInBag(phrase, feature_list=feature_list)
                if not new_phrase_item.valid:
                    continue

                similar_item_exists_in_bag = False
                for index in self.index_to_phrase_items_map.keys():
                    # print(f"{index}")
                    assert len(self.index_to_phrase_items_map[index]) > 0

                    if new_phrase_item == self.index_to_phrase_items_map[index][0]:
                        similar_item_exists_in_bag = True
                        assert self.index_to_phrase_items_map[index][0].index_in_bag is not None
                        new_phrase_item.index_in_bag = index
                        self.index_to_phrase_items_map[index].append(new_phrase_item)
                        break

                if not similar_item_exists_in_bag:
                    new_index = len(self.index_to_phrase_items_map.keys())
                    self.index_to_phrase_items_map[new_index] = [new_phrase_item]
                    new_phrase_item.index_in_bag = new_index

        self.next_available_index = len(self.index_to_phrase_items_map.keys())

        self._add_start_end_and_joker_token()

    def _add_start_end_and_joker_token(self):
        # Since part of the preprocessing is removing some too-long phrases, we'll have a joker token to fill in the
        # blanks
        self.index_to_phrase_items_map[self.next_available_index] = 'joker_token'
        self.joker_token_index = self.next_available_index
        self.next_available_index += 1

        self.index_to_phrase_items_map[self.next_available_index] = 'start_token'
        self.start_token_index = self.next_available_index
        self.next_available_index += 1

        # The end token will also represent the largest index number in the bag
        self.index_to_phrase_items_map[self.next_available_index] = 'end_token'
        self.end_token_index = self.next_available_index
        self.next_available_index += 1


class PhraseItemInBag(PhraseTensor):
    def __init__(self, phrase_object: ExtractedPhrase, feature_list):
        super().__init__(phrase_object)

        self.index_in_bag = None

        self.feature_list = feature_list

    def __eq__(self, other):
        for feature in self.feature_list:
            features = []
            exec(f"features += [self.{feature}, other.{feature}]")
            if torch.norm(features[0] - features[1]).item() != 0:
                return False
        return True


class ChoraleSentences:
    def __init__(self, bag_of_phrases: BagOfPhrases):
        self.chorale_sentence_dict: dict[str, list] = {}
        self.vocabulary_length = bag_of_phrases.next_available_index
        self.max_chorale_length = 0

        self.start_token_index = bag_of_phrases.start_token_index
        self.end_token_index = bag_of_phrases.end_token_index

        # Go over the bag of words and build the chorale sentence dictionary
        for index in bag_of_phrases.index_to_phrase_items_map.keys():
            # Iterate over all phrases in current index
            for phrase_word in bag_of_phrases.index_to_phrase_items_map[index]:
                # Skip joker, start, and end tokens
                if isinstance(phrase_word, str):
                    continue

                chorale_name = phrase_word.phrase_object.chorale_name

                # The length of each sentence will be the number of phrases plus one start token and one end token
                chorale_sentence_length = phrase_word.phrase_object.number_of_phrases_in_chorale + 2

                # Since we have a start token, the index of the phrase will be one plus its index in the original
                # chorale
                phrase_index_in_chorale_sentence = phrase_word.phrase_object.index_in_chorale + 1

                # If this is the first time we encounter this chorale sentence, initialize it into the dict with a
                # list of joker phrases, aside from the first and last elements, which will be the start and end tokens
                if chorale_name not in self.chorale_sentence_dict.keys():
                    self.chorale_sentence_dict[
                        chorale_name
                    ] = torch.ones(chorale_sentence_length) * bag_of_phrases.joker_token_index
                    self.chorale_sentence_dict[chorale_name][0] = bag_of_phrases.start_token_index
                    self.chorale_sentence_dict[chorale_name][-1] = bag_of_phrases.end_token_index

                    if chorale_sentence_length > self.max_chorale_length:
                        self.max_chorale_length = chorale_sentence_length
                        self.longest_chorale = chorale_name

                self.chorale_sentence_dict[
                    chorale_name][phrase_index_in_chorale_sentence] = phrase_word.index_in_bag

        self.number_of_chorales = len(self.chorale_sentence_dict.keys())

        # Make tensor of chorale sentences. All chorale sentences here are the same size, with the end token padding
        # the extra room.
        tensor_list = [
            torch.unsqueeze(
                torch.nn.functional.pad(
                    torch.tensor(chorale, dtype=torch.long),
                    (0, self.max_chorale_length - len(chorale)),
                    'constant',
                    chorale[-1]
                ),
                dim=0
            ) for chorale in self.chorale_sentence_dict.values()
        ]

        self.chorale_sentence_tensor = torch.concat(tensor_list, dim=0)

        self.chorale_sentence_list_of_tensors = [
            torch.tensor(chorale, dtype=torch.long) for chorale in self.chorale_sentence_dict.values()
        ]

    def phrase_index_to_onehot(self, phrase_indices):
        if isinstance(phrase_indices, int):
            phrase_indices = [phrase_indices]

        result = torch.zeros((len(phrase_indices), self.vocabulary_length), dtype=torch.int8)
        result[range(len(phrase_indices)), [phrase_index for phrase_index in phrase_indices]] = 1

        return result


