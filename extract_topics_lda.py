from datetime import datetime
import gensim
import re
import nltk
import pandas as pd
import pytz
import spacy
from gensim import corpora
from gensim.models import LdaMulticore

from features_extraction import convert_datetime_to_epoch, data_cleaning


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output


def tokenizer(texts):
    output = []
    for text in texts:
        output.append(text.split(' '))
    return output


def extract_users_topics(data_df, users, start):
    users = users.values.tolist()
    final_topic_list = []

    counter = 0
    for i in range(len(users)):
        user_data_df = data_df.loc[data_df['author'] == users[i][0]]

        if user_data_df.shape[0] == 0:
            topic_list = [0] * 21
        else:
            topic_list = extract_LDA_topics_per_user(user_data_df, start)
        final_topic_list.append(topic_list)

        counter += 1
        print('User no:', counter, 'from:', len(users), 'at', datetime.now().time())

    df = pd.DataFrame(final_topic_list)

    return df


def extract_LDA_topics_per_user(dataset, start):
    rule_epoch = int(list(filter(str.isdigit, rule))[0]) * 60 * 60
    start_epoch_date = convert_datetime_to_epoch(start)

    dataset.loc[:, 'body'] = dataset['body'].apply(data_cleaning)

    output_keywords_list = [0] * 21
    for i in range(0, 21):
        temp_dataset = dataset.loc[(dataset['created_utc'].astype(int) >= start_epoch_date + i * rule_epoch) &
                                   (dataset['created_utc'].astype(int) < start_epoch_date + (
                                           i + 1) * rule_epoch)]
        if temp_dataset.shape[0] == 0:
            continue

        text_list = temp_dataset['body'].tolist()
        tokenized_text_list = lemmatization(text_list)
        if len(tokenized_text_list[0]) == 0:
            continue

        dictionary = corpora.Dictionary(tokenized_text_list)
        doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_text_list]

        LDA = gensim.models.ldamodel.LdaModel
        lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=1, random_state=100)

        temp = lda_model.print_topics()
        topic_keywords = temp[0][1].split()  # Contains all the topic keywords
        topic_keywords = [s.replace("+", "") for s in topic_keywords]  # Remove redundant characters from LDA output
        while "" in topic_keywords:
            topic_keywords.remove("")
        keywords = []
        for k in range(len(topic_keywords)):
            keywords.append(re.sub('[^a-zA-Z]+', '', topic_keywords[k]))

        output_keywords_list[i] = keywords

    return output_keywords_list


if __name__ == "__main__":
    nltk.download('wordnet')
    rule = '8h'
    tz = pytz.timezone('Etc/GMT-3')
    start_date = datetime(2021, 6, 21, 0, 0, 0, 0, tz)

    data_df = pd.read_csv("final_dataset.csv")

    users_file_name = 'timeseries/users.txt'
    users_per_week = pd.read_csv(users_file_name, header=None).transpose()

    lda_df = extract_users_topics(data_df, users_per_week, start_date)
    lda_df.to_csv("lda_topics/users_lda_topics.csv", header=True, index=False)

    print("Debugging Message")
