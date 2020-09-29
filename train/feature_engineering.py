from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scripts.train.decorator import Decorator
decorator = Decorator()
import pandas as pd

class FeatureExtraction:

    def featrue_extract(self,df1):
        #cvz = CountVectorizer()
        #cvz.fit(df1["cleaned"].values)
        #count_vectors = cvz.transform(df1["cleaned"].values)

       #word_tfidf = TfidfVectorizer(analyzer='word', max_features = 5000)
        word_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                                ngram_range=(1, 2),
                                stop_words='english')
        word_tfidf.fit(df1["cleaned"].values)
        word_vectors_tfidf = word_tfidf.transform(df1["cleaned"].values)
        return word_tfidf,word_vectors_tfidf

    def label_encoding(self,df1):
        target = df1["Category"].values
        target = LabelEncoder().fit_transform(target)
        return target

    def df_with_targets_actuals(self, df,target):
        df1 = pd.DataFrame(target, columns=['encoded_categories'])
        df1.reset_index(level=0, inplace=True)
        x = df.reset_index(level=0)
        df_all = x.merge(df1, on='index', indicator=True)
        df_all.reset_index(drop=True,inplace=True)
        df_all = df_all[['encoded_categories','Category']].drop_duplicates()
        # Dictionaries for future use
        id_to_category = dict(df_all[['encoded_categories', 'Category']].values)
        return id_to_category