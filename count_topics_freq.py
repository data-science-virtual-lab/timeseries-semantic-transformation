import pandas as pd
import re
import glob
from nltk.corpus import wordnet


def count_keywords_frequency(k):
    all_files = glob.glob("lda_topics/users_lda_topics.csv")
    all_files_df = pd.concat([pd.read_csv(f) for f in all_files])

    words_df = []
    for i in range(all_files_df.shape[0]):
        for j in range(all_files_df.shape[1]):
            temp = all_files_df.iloc[i, j]
            if (temp != '0') and (type(temp) == str):
                temp = re.sub('[^a-zA-Z]+', ' ', temp)
                temp = ' '.join(temp.split())
                temp = list(temp.split(" "))
                for word in temp:
                    words_df.append(word)
    words_df = pd.DataFrame(words_df)

    frequent_words = words_df[0].value_counts()
    frequent_words = frequent_words.to_frame()
    frequent_words = frequent_words.iloc[0:, :]
    frequent_words.reset_index(level=0, inplace=True)
    frequent_words.columns = ['keyword', 'frequency']

    return frequent_words['keyword'][0:k].tolist()


def find_synonyms(word_list):
    all_words_synonyms = []
    for word in word_list:
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        synonyms = list(dict.fromkeys(synonyms))
        all_words_synonyms.extend(synonyms)

    return all_words_synonyms


def find_freq_words_expanded(k):
    freq_words = count_keywords_frequency(k)
    freq_words.extend(find_synonyms(freq_words))
    freq_words_expanded = list(set(freq_words))

    return freq_words_expanded


if __name__ == "__main__":
    k = 10
    keywords = find_freq_words_expanded(k)

    print("Debugging Message")
