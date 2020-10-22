import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *


ratings_ds = pd.read_csv(DATASET_PATH)

# print(ratings_ds.head())

# ratings = ratings_ds['RATING'].unique()

ratings_counts = ratings_ds['RATING'].value_counts(ascending=True)
print(ratings_counts)

fig, ax = plt.subplots()
ratings_counts.plot(ax=ax, kind='bar', title="Balance of ratings", xlabel="Ratings", ylabel="Frecuency", rot=0)
plt.savefig(f"{OUTPUT_PATH}/Balance_of_classes.png")
