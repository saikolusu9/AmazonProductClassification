from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class Decorator:

    def def_decorator(self, f):
        def wrapper(*args,**kwargs):
            print("The function entered: " + f.__name__)
            x = f(*args,**kwargs)
            return x

        return wrapper