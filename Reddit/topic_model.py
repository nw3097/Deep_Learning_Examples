from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
import pandas as pd
import os
home = os.getenv('HOME')

class TopicModel(object):

    def __init__(self):

        self.data = self._get_data()

    def _get_data(self):

        data = pd.read_csv(home + '/Deep_Learning_Examples/Reddit/cumulative_data.csv')
        data = data.dropna(subset=['selftext'],axis=0)
        data.loc[:, 'selftext'] = data.selftext.apply(lambda x: x.replace('\n', '').lstrip().rstrip().lower())
        return data

    def lda(self):

        sw = stopwords.words('english')
        tokenizer = RegexpTokenizer(r'\w+')
        stemmer = PorterStemmer()

        posts = []
        for post in self.data.selftext:
            tokens = tokenizer.tokenize(post)
            stopped_tokens = [x for x in tokens if not x in sw]
            stemmed_tokens = [stemmer.stem(x) for x in stopped_tokens]

            posts.append(stemmed_tokens)

        dictionary = corpora.Dictionary(posts)
        corpus = [dictionary.doc2bow(post) for post in posts]

        clf = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=7, id2word=dictionary, passes=10)
        clf.save(home + '/Deep_Learning_Examples/Reddit/model_07_topics_10_passes')

        with open('corpus.pkl', 'wb') as fp:
            pickle.dump(corpus, fp)

        with open('dictionary.pkl', 'wb') as fp:
            pickle.dump(dictionary, fp)

        print("break")

if __name__ == '__main__':

    tm = TopicModel()
    tm.lda()
    print('break')
