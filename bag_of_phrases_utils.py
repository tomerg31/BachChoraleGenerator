from copy import deepcopy
import os
import pickle
from bag_of_phrase_words import ChoraleSentences, BagOfPhrases


def make_feature_combinations():
    # Make different bags of phrases by filtering with different feature combinations

    # All features in this function are from PHRASE_FEATURE_LIST in chorale_phrase_tensor.py

    # A mapping from a creative, informative name, to a list of features
    feature_combinations = {}

    basic_two_harmonies = [
        'opening_downbeat_harmony',
        'fermata_harmony_inversion'
    ]

    basic_three_harmonies_opening = [
        'opening_pickup_harmony',
        'opening_downbeat_harmony',
        'fermata_harmony_inversion',
    ]

    basic_three_harmonies_closing = [
        'opening_downbeat_harmony',
        'pre_fermata_harmony',
        'fermata_harmony_inversion'
    ]

    basic_all_harmonies = [
        'opening_pickup_harmony',
        'opening_downbeat_harmony',
        'pre_fermata_harmony',
        'fermata_harmony_inversion'
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


