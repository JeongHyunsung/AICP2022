import spacy


def tokenizer_spacy(sentence):
    nlp = spacy.load("en_core_web_sm")
    return [token.text for token in nlp(sentence)]