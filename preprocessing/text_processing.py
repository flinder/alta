import spacy
import pandas as pd
import Stemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class TextProcessor(object):

    def __init__(self, language='en'):
        self.parser = spacy.load(language)
        self.stemmer = Stemmer.Stemmer(language)

    def _lemmatize(self, text):
        parsed = self.parser(text)
        return ' '.join([x.lemma_ for x in parsed])

    def _stem(self, text):
        parsed = self.parser(text)
        return ' '.join([self.stemmer.stemWord(x.orth_) for x in parsed])

    def vectorize(self, texts, tfidf, stem, token_type_gram_size):
        token_type, gram_size = token_type_gram_size
        if tfidf:
            Vectorizer = TfidfVectorizer
        else:
            Vectorizer = CountVectorizer
        if stem:
            preproc = self._stem
        else:
            preproc = self._lemmatize

        vectorizer = Vectorizer(preprocessor=preproc, analyzer=token_type, 
                                ngram_range=(gram_size,gram_size))

        return vectorizer.fit_transform(texts)


if __name__ == "__main__":
    '''
    Test
    '''

    text = pd.Series(["hello I'm text one. with some stuff \n in here",
                      ('text 2 here. testing the stemming. testing the '
                      'lemmatization. I dove into the pool. The dove jumped into'
                      ' the pool 999 times.')])

    text_processor = TextProcessor('en')

    output = text_processor.vectorize(text)
