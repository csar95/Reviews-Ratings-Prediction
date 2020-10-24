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
VOCABULARY_SIZE = 5200
MAX_REVIEW_LENGTH = 100
EMBEDDING_DIM = 32
NUM_EPOCHS = 75
####################

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

def plot_loss_acc(epochs, train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)

    plt.plot(epochs, train_loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'b--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(epochs, train_acc, 'g-', label='Training acc')
    plt.plot(epochs, val_acc, 'g--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(f"{OUTPUT_PATH}/Train_vs_Validation_DS.png")
