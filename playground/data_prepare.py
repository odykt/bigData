import pandas as pd
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from misc import *



file_path = '../bigdata2025classification/train.csv'
dataTrain = load_and_process_data(file_path)

check_duplicates(dataTrain)
check_title_duplicates(dataTrain)
check_content_duplicates(dataTrain)
check_title_content_duplicates(dataTrain)
check_column_types(dataTrain)

# Remove duplicates based on 'Title' and 'Content' columns, keeping the first occurrence
dataTrain = dataTrain.drop_duplicates(subset=['Title', 'Content'], keep='first')
print("\nDuplicates based on Title and Content removed. Data shape:", dataTrain.shape)

# Reset the index after removing duplicates
dataTrain = dataTrain.reset_index(drop=True)
print("\nIndex reset. Data shape:", dataTrain.shape)
dataTrain.info()

# Download required NLTK data if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


print("Stem and lemmatize\n\n")
for col in ['Title', 'Content']:
    dataTrain[col] = dataTrain[col].astype(str).apply(clean_text,args=(stop_words,stemmer,lemmatizer,2))

print("get statistics")

dataTrain['text_length'] = dataTrain['Content'].apply(len)
dataTrain['word_count'] = dataTrain['Content'].apply(lambda x: len(str(x).split()))
dataTrain['sentence_count'] = dataTrain['Content'].apply(lambda x: len(str(x).split('.')))
dataTrain['avg_word_length'] = dataTrain['Content'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))


print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())

import joblib
joblib.dump(dataTrain,"data_train.obj")
