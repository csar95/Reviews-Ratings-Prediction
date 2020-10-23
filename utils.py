import matplotlib.pyplot as plt
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



#################### PARAMETERS

DATASET_PATH = "./Resources/ratings.csv"
OUTPUT_PATH = "./Output"


def plot_class_balance(data, title, filename):
    fig, ax = plt.subplots()
    data.plot(ax=ax, kind='bar', title=title, xlabel="Ratings", ylabel="Frecuency", rot=0)
    plt.savefig(f"{OUTPUT_PATH}/{filename}.png")

def text_preprocessing(text):
    # Lowercase the text to reduce the vocabulary of our data
    text = text.lower()

    # Remove numbers. Not necessary to convert them into text
    text = re.sub(r'\d+', '', text)

    # Remove punctuations so we don’t have different forms of the same word
    text = text.translate( str.maketrans('', '', string.punctuation) )

    # Remove words that do not contribute to the meaning of a sentence (stop-words)
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    text_tokens = [word for word in word_tokens if word not in stop_words]

    # Apply stemming to reduce words to their root form
    # stemmer = PorterStemmer()
    # text_tokens = [stemmer.stem(word) for word in text_tokens]

    # Apply lemmatization to reduce words to their root form ensuring that the root word belongs to the language
    lemmatizer = WordNetLemmatizer()
    text_tokens = [lemmatizer.lemmatize(word, pos ='v') for word in text_tokens]

    # Remove all the white spaces in a string
    text = " ".join(text_tokens)

    return text
