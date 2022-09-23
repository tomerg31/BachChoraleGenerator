from copy import deepcopy
import os
import pickle
from bag_of_phrase_words import ChoraleSentences, BagOfPhrases
import torch


def make_feature_combinations():
    # Make different bags of phrases by filtering with different feature combinations

    # All features in this function are from PHRASE_FEATURE_LIST in chorale_phrase_tensor.py

    # A mapping from a creative, informative name, to a list of features
    feature_combinations = {}

    basic_two_harmonies = [
        'opening_downbeat_harmony',
        'fermata_harmony'
    ]

    basic_three_harmonies_opening = [
        'opening_pickup_harmony',
        'opening_downbeat_harmony',
        'fermata_harmony',
    ]

    basic_three_harmonies_closing = [
        'opening_downbeat_harmony',
        'pre_fermata_harmony',
        'fermata_harmony'
    ]

    basic_all_harmonies = [
        'opening_pickup_harmony',
        'opening_downbeat_harmony',
        'pre_fermata_harmony',
        'fermata_harmony'
    ]

    feature_combinations['basic_two_harmonies'] = basic_two_harmonies
    feature_combinations['basic_three_harmonies_opening'] = basic_three_harmonies_opening
    feature_combinations['basic_three_harmonies_closing'] = basic_three_harmonies_closing
    feature_combinations['basic_all_harmonies'] = basic_all_harmonies

    # Add combinations as above that include the mode feature
    new_combinations = {}
    for combination_name in feature_combinations.keys():
        new_feature_combination = deepcopy(feature_combinations[combination_name])
        new_feature_combination += ['phrase_mode']
        new_combination_name = 'mode' + combination_name[5:]
        new_combinations[new_combination_name] = new_feature_combination

    feature_combinations.update(new_combinations)

    return feature_combinations


def make_bags_of_phrases_from_feature_combinations(
        feature_combinations,
        bags_of_phrases_directory_name='bags_of_phrases'
):
    try:
        os.makedirs(bags_of_phrases_directory_name, exist_ok=False)
    except FileExistsError:
        pass

    with open("extracted_chorale_container.pkl", 'rb') as extracted_chorale_container_pkl:
        extracted_chorale_container = pickle.load(extracted_chorale_container_pkl)

    for combination_name in feature_combinations.keys():
        bag_of_phrases_feature_combination_full_path = os.path.join(
            bags_of_phrases_directory_name,
            combination_name + '.pkl'
        )

        print("Processing", bag_of_phrases_feature_combination_full_path)

        if os.path.exists(bag_of_phrases_feature_combination_full_path):
            print(bag_of_phrases_feature_combination_full_path, "exists, skipping")
            continue

        print(bag_of_phrases_feature_combination_full_path, "does not exist, creating and saving to pkl file")

        bag_of_phrases = BagOfPhrases(extracted_chorale_container, feature_combinations[combination_name])

        with open(bag_of_phrases_feature_combination_full_path, 'wb') as bag_of_phrases_pkl:
            pickle.dump(bag_of_phrases, bag_of_phrases_pkl)


def make_chorale_sentences_by_feature_combination(
        chorale_sentences_by_feature_combination_pickle_filename,
        bag_of_phrases_directory='bags_of_phrases'
):
    chorale_sentences_by_feature_combination = {}

    for bag_of_phrase_filename in os.listdir(bag_of_phrases_directory):
        with open(os.path.join(bag_of_phrases_directory, bag_of_phrase_filename), 'rb') as bag_of_phrases_pkl:
            # We assume that the filename ends with .pkl
            chorale_sentences_by_feature_combination[bag_of_phrase_filename[:-4]] = \
                ChoraleSentences(pickle.load(bag_of_phrases_pkl))

    # Save chorale_sentences_by_feature_combination to a pkl file
    with open('chorale_sentences_by_feature_combination.pkl', 'wb') as chorale_sentences_by_feature_combination_pkl:
        pickle.dump(chorale_sentences_by_feature_combination, chorale_sentences_by_feature_combination_pkl)


def chorale_sentences_tensor_to_labelled_samples(chorale_sentences, device):
    # If train with batches is true, we return a tensor, if not we return lists of tensors

    embedded_chorale_sentences = torch.nn.functional.one_hot(chorale_sentences.chorale_sentence_tensor).to(device)

    samples = embedded_chorale_sentences[:, :-1, :]
    labels = chorale_sentences.chorale_sentence_tensor[:, 1:].to(device)

    return samples, labels


def chorale_sentences_list_to_labelled_samples(chorale_sentences: ChoraleSentences):
    samples, labels = [], []
    for chorale_sentence in chorale_sentences.chorale_sentence_list_of_tensors:
        samples.append(chorale_sentence[:-1])
        labels.append(chorale_sentence[1:])

    return samples, labels


class PhraseCollator(object):
    def __init__(self, chorale_sentences: ChoraleSentences, device):
        self.chorale_sentences = chorale_sentences
        self.device = device

    def __call__(self, batch):
        samples, lengths, labels = [], [], []

        for (sample, label) in batch:
            samples.append(torch.tensor(sample))
            lengths.append(len(sample))
            labels.append(torch.tensor(label))

        samples = torch.nn.utils.rnn.pad_sequence(
            samples,
            batch_first=True,
            padding_value=self.chorale_sentences.pad_token_index
        )

        samples = torch.nn.functional.one_hot(samples, num_classes=self.chorale_sentences.vocabulary_length)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.chorale_sentences.pad_token_index
        )

        return samples.to(self.device), lengths, labels.to(self.device)

