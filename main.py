import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk

from nltk.tokenize import word_tokenize
from collections import Counter
from utils import *

#################### LOAD DATA
ratings_ds = pd.read_csv(DATASET_PATH)
print(ratings_ds.head())

# ratings = ratings_ds['RATING'].unique()

#################### DATA ANALYSIS & DATA CLEANING
# ratings_counts = ratings_ds['RATING'].value_counts(ascending=True)
# plot_class_balance(ratings_counts, title="Balance of ratings", filename="Balance_of_classes")

ratings_ds_preprocess = ratings_ds.drop_duplicates(ignore_index=True)  # Drop duplicate rows keeping the first appearance and reindexing
print(ratings_ds_preprocess.head())

# ratings_counts = ratings_ds_preprocess['RATING'].value_counts()
# plot_class_balance(ratings_counts, title="Balance of ratings (without duplicates)", filename="Balance_of_classes_no_dupl")

# print(ratings_ds_preprocess.isnull().sum())  # Look for null values
ratings_ds_preprocess = ratings_ds_preprocess.dropna()  # Drop all the rows in which any null value is present

ratings_ds_preprocess['TEXT'] = ratings_ds_preprocess['TEXT'].apply(text_preprocessing)  # Apply preprocessing to every review
print(ratings_ds_preprocess.head())

# print(ratings_ds_preprocess[ratings_ds_preprocess['TEXT'] == ""])  # Look for empty strings
ratings_ds_preprocess['TEXT'].replace("", np.nan, inplace=True)  # Replace empty str by NaN so we can drop them
# print(ratings_ds_preprocess.isnull().sum())
ratings_ds_preprocess = ratings_ds_preprocess.dropna()  # Drop all the rows in which any null value is present

# ratings_counts = ratings_ds_preprocess['RATING'].value_counts()
# plot_class_balance(ratings_counts, title="Balance of ratings (after cleaning)", filename="Balance_of_classes_aft_clean")

most_common_words_per_rating = ratings_ds_preprocess.groupby('RATING')['TEXT'].apply(lambda review: Counter(" ".join(review).split()).most_common(5))
most_common_words_per_rating.to_csv(f"./{OUTPUT_PATH}/most_common_words_per_rating.txt")

