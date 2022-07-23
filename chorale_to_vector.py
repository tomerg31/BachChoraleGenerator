from abstract_chorale_objects import ChoraleVector
import music21
from tqdm import tqdm
import pickle

chorales_iterator = music21.corpus.chorales.Iterator()
transformed_chorales = []

for chorale in tqdm(chorales_iterator):
    if len(chorale.parts) > 4:
        continue

    transformed_chorales.append(ChoraleVector(chorale))

b = 1

with open('vectorized_chorales', 'wb') as vectorized_chorale_file:
    pickle.dump(transformed_chorales, vectorized_chorale_file)




