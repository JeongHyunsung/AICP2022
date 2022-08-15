import spacy
import nltk


class tokenizer_spacy:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, sentence):
        return [token.text for token in self.nlp(sentence)]


class tokenizer_nltk:
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
