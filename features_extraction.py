import csv
import re
import string
import numpy as np
from Levenshtein import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob


def convert_datetime_to_epoch(date_time):
    epoch_time = date_time.timestamp()

    return epoch_time


def extract_features(data_df):
    data_df.insert(2, "commitment", 1)
    data_df.insert(3, "sentiment", 0.0)
    data_df.insert(4, "product_mix", 0.0)

    data_df = calculate_polarity_product_mix(data_df)

    return data_df


def data_cleaning(text):
    text = re.sub(r'RT @[a-z,A-Z,0-9,_]*: ', ' ', text)
    text = re.sub(r'RT:', ' ', text)
    text = re.sub(r'@[a-z,A-Z,0-9,_]*', ' ', text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text)
    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))  # strip punctuation
    text = " ".join(text.split())  # removes all whitespace (spaces, tabs, newlines) from anywhere in sentence

    text = text.lower()

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [w for w in words if not w in stop_words]  # stopwords removal

    text = " ".join(words)

    return text


def calculate_polarity_product_mix(df):
    file_path = 'lexicons\\starbucks_products_lexicon.csv'
    with open(file_path, newline='', encoding="utf8") as f:
        products_list = list(csv.reader(f))
    products_list = products_list[0]

    prods = list()
    for products in products_list:
        prods = prods + products.split(' ')
    products = list(set(prods))

    i = 0
    for index, row in df.iterrows():
        user_tweets = row['body']
        user_tweets = data_cleaning(user_tweets)

        num_of_products = 0
        user_text = user_tweets.split(' ')
        for w in user_text:
            for product in products:
                if distance(w, product) <= 1:
                    num_of_products += 1
                    break

        df.loc[index, 'product_mix'] = num_of_products

        polarity = TextBlob(user_tweets).polarity
        df.loc[index, 'sentiment'] = polarity

        i = i + 1
        print("Have processed", i, "post(s) out of a total of", len(df))
    return df


def remove_non_active_users(df):
    users_activity = df.groupby('author').size().tolist()
    th = np.percentile(users_activity, 99.99)
    print("Threshold: ", th)
    df = df.groupby('author').filter(lambda x: len(x) <= th)
    df = df.groupby('author').filter(lambda x: len(x) > 1)
    return df
