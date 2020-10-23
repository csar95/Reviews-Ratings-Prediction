import pandas as pd
import numpy as np
import tensorflow as tf

from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from utils import *

#################### LOAD DATA
ratings_df = pd.read_csv(DATASET_PATH)
ratings = ratings_df['RATING'].unique()

#################### DATA ANALYSIS & DATA CLEANING
# ratings_counts = ratings_df['RATING'].value_counts(ascending=True)
# plot_class_balance(ratings_counts, title="Balance of ratings", filename="Balance_of_classes")

ratings_df_preprocess = ratings_df.drop_duplicates(ignore_index=True)  # Drop duplicate rows keeping the first appearance and reindexing

# ratings_counts = ratings_df_preprocess['RATING'].value_counts()
# plot_class_balance(ratings_counts, title="Balance of ratings (without duplicates)", filename="Balance_of_classes_no_dupl")

# print(ratings_df_preprocess.isnull().sum())  # Look for null values
ratings_df_preprocess = ratings_df_preprocess.dropna()  # Drop all the rows in which any null value is present

ratings_df_preprocess['TEXT'] = ratings_df_preprocess['TEXT'].apply(text_preprocessing)  # Apply preprocessing to every review

# print(ratings_df_preprocess[ratings_df_preprocess['TEXT'] == ""])  # Look for empty strings
ratings_df_preprocess['TEXT'].replace("", np.nan, inplace=True)  # Replace empty str by NaN so we can drop them
# print(ratings_df_preprocess.isnull().sum())
ratings_df_preprocess = ratings_df_preprocess.dropna()  # Drop all the rows in which any null value is present

# ratings_counts = ratings_df_preprocess['RATING'].value_counts()
# plot_class_balance(ratings_counts, title="Balance of ratings (after cleaning)", filename="Balance_of_classes_aft_clean")

# most_common_words_per_rating = ratings_df_preprocess.groupby('RATING')['TEXT'].apply(lambda review: Counter(" ".join(review).split()).most_common(5))
# most_common_words_per_rating.to_csv(f"./{OUTPUT_PATH}/most_common_words_per_rating.txt")

#################### PREPARE THE DATASET
ratings_df_preprocess['RATING'] = ratings_df_preprocess['RATING'] -1

train_df, test_df = train_test_split(ratings_df_preprocess, test_size=0.1)
print(test_df['RATING'].value_counts())
train_df, val_df = train_test_split(train_df, test_size=0.1)
print(train_df['RATING'].value_counts())
print(val_df['RATING'].value_counts())

target = train_df.pop('RATING')
raw_train_ds = tf.data.Dataset.from_tensor_slices((train_df.values, target.values))
target = test_df.pop('RATING')
raw_test_ds = tf.data.Dataset.from_tensor_slices((test_df.values, target.values))
target = val_df.pop('RATING')
raw_val_ds = tf.data.Dataset.from_tensor_slices((val_df.values, target.values))

vectorize_layer = layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    output_mode='int',  # Create unique integer indices for each token
    output_sequence_length=MAX_REVIEW_LENGTH)

def vectorize_text(review, rating):
    review = tf.expand_dims(review, -1)
    return vectorize_layer(review), rating

# Analyze the dataset, determine the frequency of individual string values, and create a 'vocabulary' from them
vectorize_layer.adapt(raw_train_ds.map(lambda review, _: review))
print(vectorize_layer.get_vocabulary())

# Apply the TextVectorization layer to each dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#################### CREATE THE MODEL

model = tf.keras.Sequential([
    # This layer takes the integer-encoded text and looks up an embedding vector for each word-index. These vectors are learned as the model trains.
    layers.Embedding(input_dim=VOCABULARY_SIZE + 1,  # Size of the vocabulary (i.e. maximum integer index + 1)
                     output_dim=EMBEDDING_DIM,
                     input_length=MAX_REVIEW_LENGTH),  # Length of input sequences, when it is constant
    layers.Dropout(0.2),

    # For each feature dimension, this layer takes average among all time steps
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),

    # Fully connected layer
    layers.Dense(units=ratings.size)
])

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

print(model.summary())

#################### TRAIN AND EVALUATE THE MODEL

history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

plot_loss_acc(range(1, NUM_EPOCHS +1), history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])
