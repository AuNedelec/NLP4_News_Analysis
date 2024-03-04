import codecademylib3_seaborn
import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

# import CountVectorizer, TfidfTransformer, TfidfVectorizer for our analysis
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer

# view random article to get an insight into what some of them are about manually
print(articles[7])

# preprocess articles
processed_articles = [preprocess_text(doc) for doc in articles]
print(processed_articles[3])

# initialize and fit CountVectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(processed_articles)

# convert counts to tf-idf
transformer = TfidfTransformer(norm=None)
tfidf_scores_transformed = transformer.fit_transform(counts)

# initialize and fit TfidfVectorizer to verify that results of Transfomer are the same as using directly Vectorizer
vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(processed_articles)

# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

# get highest scoring tf-idf term for each article
for i in range(1, 10):
  print(df_tf_idf[[f'Article{i}']].idxmax())
