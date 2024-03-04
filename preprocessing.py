import nltk, re  # Importing necessary libraries
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Get English stop words
stop_words = stopwords.words('english')

# Initialize WordNetLemmatizer
normalizer = WordNetLemmatizer()

# Function to determine the part of speech of a word
def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)  # Get probable part of speech using WordNet
  pos_counts = Counter()  # Initialize a Counter to count different parts of speech
  # Count nouns, verbs, adjectives, and adverbs
  pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
  pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
  pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
  pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
  # Get the most common part of speech
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech  # Return the most likely part of speech

# Function to preprocess text
def preprocess_text(text):
  cleaned = re.sub(r'\W+', ' ', text).lower()  # Remove non-alphanumeric characters and convert to lowercase
  tokenized = word_tokenize(cleaned)  # Tokenize the text
  # Lemmatize tokens based on their part of speech and remove digits
  normalized = " ".join([normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized if not re.match(r'\d+',token)])
  return normalized  # Return the preprocessed text
