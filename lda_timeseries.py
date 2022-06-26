import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import pytz
from Levenshtein import distance

from count_topics_freq import find_freq_words_expanded
from features_extraction import convert_datetime_to_epoch, extract_features


def algorithm2(keywords, users_lda):
    commitment_numpy = np.zeros(shape=(users_lda.shape[0], 21))
    sentiment_numpy = np.zeros(shape=(users_lda.shape[0], 21))
    product_mix_numpy = np.zeros(shape=(users_lda.shape[0], 21))
    for i in range(users_lda.shape[0]):
        for k in range(0, 21):
            temp_list = users_lda[i, k]
            if temp_list == 0 or temp_list == '0':
                continue
            temp_list = temp_list.split(",")
            topics_temp = []
            for item in temp_list:
                item = ''.join(e for e in item if e.isalnum())
                topics_temp.append(item)
            count = 0
            for keyword in keywords:
                for item in topics_temp:
                    if distance(keyword, item) <= 1:
                        count += 1
            if count != 0:
                comm, sent, prod = calculateBehavior(i, k)
                commitment_numpy[i, k + 1] = comm
                sentiment_numpy[i, k + 1] = sent
                product_mix_numpy[i, k + 1] = prod

    return commitment_numpy, sentiment_numpy, product_mix_numpy


def calculateBehavior(i, k):
    start_date = datetime(2021, 6, 21, 0, 0, 0, 0, tz)
    start_epoch_date = convert_datetime_to_epoch(start_date)
    data = pd.read_csv("final_dataset.csv")

    day_start = int(start_epoch_date + k * 8 * 60 * 60)
    day_end = int(start_epoch_date + (k + 1) * 8 * 60 * 60)

    temp_data = data[
        (data['created_utc'] > day_start) & (data['created_utc'] < day_end) & (data['author'] == 'user' + str(i + 1))]
    temp_data = extract_features(temp_data)

    return temp_data['commitment'].sum(), temp_data['sentiment'].sum(), temp_data['product_mix'].sum()


if __name__ == "__main__":

    starbucks_account_keywords = ['beloved', 'cherished', 'favored', 'main', 'popular', 'prized', 'treasured', 'choice',
                                  'darling',
                                  'dear', 'dearest', 'intimate', 'personal', 'pet', 'sweetheart', 'admired', 'adored',
                                  'best-loved',
                                  'desired', 'especial', 'esteemed', 'liked', 'number one', 'pleasant', 'precious',
                                  'revered',
                                  'wished-for', 'imbibe', 'quaff', 'savor', 'extract', 'partake', 'sample', 'sup',
                                  'swallow', 'taste',
                                  'toss', 'drink in', 'afford', 'cater', 'contribute', 'deliver', 'equip', 'feed',
                                  'fill', 'find',
                                  'grant', 'hand over', 'produce', 'store', 'transfer', 'turn over', 'yield',
                                  'dispense',
                                  'drop', 'endow',
                                  'fulfill', 'hand', 'heel', 'minister', 'outfit', 'provision', 'purvey', 'replenish',
                                  'satisfy', 'stake',
                                  'stock', 'victual', 'cater to', 'come across with', 'come through', 'come up with',
                                  'fix up',
                                  'give with', 'kick in', 'pony up', 'put out', 'put up', 'conglomerate', 'group',
                                  'string',
                                  'alternation', 'catena', 'concatenation', 'continuity', 'order', 'progression', 'row',
                                  'sequence',
                                  'set', 'syndicate', 'train', 'trust', 'consecution', 'dealer', 'hawker', 'merchant',
                                  'peddler',
                                  'businessperson', 'huckster', 'pitcher', 'traveler', 'outcrier', 'composure',
                                  'diligence', 'endurance',
                                  'fortitude', 'grit', 'humility', 'moderation', 'perseverance', 'persistence', 'poise',
                                  'restraint',
                                  'self-control', 'tolerance', 'backbone', 'bearing', 'calmness', 'constancy', 'cool',
                                  'equanimity',
                                  'forbearance', 'guts', 'gutsiness', 'heart', 'imperturbability', 'legs', 'leniency',
                                  'long-suffering',
                                  'longanimity', 'moxie', 'nonresistance', 'passiveness', 'passivity', 'resignation',
                                  'serenity',
                                  'starch', 'stoicism', 'submission', 'sufferance', 'toleration', 'yielding',
                                  'even temper',
                                  'intestinal fortitude', 'staying power', 'acquaintance', 'ally', 'associate', 'buddy',
                                  'classmate',
                                  'colleague', 'companion', 'cousin', 'partner', 'roommate', 'chum', 'cohort',
                                  'compatriot', 'comrade',
                                  'consort', 'crony', 'familiar', 'intimate', 'mate', 'pal', 'playmate', 'schoolmate',
                                  'sidekick',
                                  'spare', 'well-wisher', 'alter ego', 'bosom buddy', 'soul mate']

    users_topics = pd.read_csv("lda_topics/users_lda_topics.csv")
    users_topics = users_topics.to_numpy()

    tz = pytz.timezone('Etc/GMT-3')
    start_date = datetime(2021, 6, 21, 0, 0, 0, 0, tz)

    commitment, sentiment, product_mix = algorithm2(starbucks_account_keywords, users_topics)

    commitment = pd.DataFrame(commitment)
    sentiment = pd.DataFrame(sentiment)
    product_mix = pd.DataFrame(product_mix)

    commitment.insert(0, "user", ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8'], True)
    sentiment.insert(0, "user", ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8'], True)
    product_mix.insert(0, "user", ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8'], True)

    commitment.to_csv('final_timeseries\\' + 'user_timeseries_commitment.txt', index=False, header=None)
    sentiment.to_csv('final_timeseries\\' + 'user_timeseries_sentiment.txt', index=False, header=None)
    product_mix.to_csv('final_timeseries\\' + 'user_timeseries_product_mix.txt', index=False, header=None)

    print("Debugging Message")
