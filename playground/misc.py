import pandas as pd
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
import re

def load_and_process_data(file_path):
    # Load data from a CSV file
    dataTrain = pd.read_csv(file_path)

    print("Data loaded successfully.")
    print("First 5 rows of the dataset:")
    print(dataTrain.head())

    print("\nData summary:")
    print(dataTrain.info())

    # Check for missing values in the dataframe
    print("\nMissing values in each column:")
    print(dataTrain.isnull().sum())
    
    return dataTrain

# check column data types
def check_column_types(dataTrain):
    print("\nColumn data types:")
    print(dataTrain.dtypes)


# Check for duplicate rows in the dataframe
def check_duplicates(dataTrain):
    duplicate_count = dataTrain.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    return duplicate_count

# Check for duplicates based only on 'Title' column
def check_title_duplicates(dataTrain):
    if 'Title' in dataTrain.columns:
        dup_count = dataTrain.duplicated(subset=['Title']).sum()
        print(f"\nNumber of duplicate rows based on Title: {dup_count}")
        return dup_count
    else:
        print("'Title' column not found in the dataframe.")
        return None

# Check for duplicates based only on 'Content' column
def check_content_duplicates(dataTrain):
    if 'Content' in dataTrain.columns:
        dup_count = dataTrain.duplicated(subset=['Content']).sum()
        print(f"\nNumber of duplicate rows based on Content: {dup_count}")
        return dup_count
    else:
        print("'Content' column not found in the dataframe.")
        return None


# Check for duplicates based on 'Title' and 'Content' columns
def check_title_content_duplicates(dataTrain):
    if 'Title' in dataTrain.columns and 'Content' in dataTrain.columns:
        dup_count = dataTrain.duplicated(subset=['Title', 'Content']).sum()
        print(f"\nNumber of duplicate rows based on Title and Content: {dup_count}")
        return dup_count
    else:
        print("'Title' and/or 'Content' columns not found in the dataframe.")
        return None


#def clean_text(text, stop_words, stemmer, lemmatizer):
#    # Expand contractions
#    text = contractions.fix(text)
#    # Convert to lowercase
#    text = text.lower()
#    # Remove special characters (keep only letters and spaces)
#    text = re.sub(r'[^a-z\s]', '', text)
#    # Remove extra spaces
#    text = re.sub(r'\s+', ' ', text).strip()
#    # Tokenize
#    words = text.split()
#    # Remove stopwords, lemmatize, and stem
#    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
#    text = ' '.join(words)
#    return text
#

def clean_text(text,stop_words,stemmer=None,lemmatizer=None,temperature=3):
    """
    temperature:
        1 -> stopwords only
        2 -> stopwords + lemmatization
        3 -> stopwords + lemmatization + stemming
    """

    if not isinstance(text, str):
        return ""

    # Expand contractions (don't skip this – it matters)
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Remove non-letter characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize (simple whitespace split)
    tokens = text.split()

    # Stopword removal (always applied)
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatization (temperature >= 2)
    if temperature >= 2:
        if lemmatizer is None:
            raise ValueError("lemmatizer must be provided when temperature >= 2")
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Stemming (temperature == 3)
    if temperature >= 3:
        if stemmer is None:
            raise ValueError("stemmer must be provided when temperature >= 3")
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

