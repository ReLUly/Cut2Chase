import networkx
from collections import Counter
from itertools import combinations
from konlpy.tag import Okt
import re


def summarize(document, unit='sentence', lines_to_summarize=3, language='kor'):
    class Sentence:
        def __init__(self, sentence, index):
            kkma = Okt()
            self.sentence = sentence
            self.nouns = kkma.nouns(self.sentence)
            self.index = index

        def __eq__(self, other):
            return isinstance(other, Sentence) and other.index == self.index

        def __hash__(self):
            return self.index

        def __str__(self):
            return self.sentence

    class Word:
        def __init__(self, word, index):
            self.word = word
            self.index = index

        def __eq__(self, other):
            return isinstance(other, Word) and other.index == self.index

        def __hash__(self):
            return self.index

        def __str__(self):
            return self.word

    def segment(text):
        corpus = re.split('(?<!\d)\.|\.(?!\d)', text)
        result = []
        for sentence in corpus:
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        return result

    def make_vocab(text):
        twitter = Okt()
        vocab = []
        for sentence in text:
            word = twitter.morphs(sentence)
            for w in word:
                vocab.append(w)
        return list(set(vocab))

    doc = [Sentence(stc, i) for i, stc in enumerate(segment(document))]
    wrd = [Word(wrd, i) for i, wrd in enumerate(make_vocab(document))]
    sen = segment(document)

    def jac_index(a: Sentence, b: Sentence):
        mult_a = Counter(a.nouns)
        mult_b = Counter(b.nouns)
        if sum((mult_a | mult_b).values()) == 0:
            return 0
        return sum((mult_a & mult_b).values()) / sum((mult_a | mult_b).values())

    def tf_idf(a, b):
        pass

    def freq(a, b):
        result = 0
        for sentence in sen:
            if a.word in sentence and b.word in sentence:
                result += 1
        return result/sum([
            a.word in line or b.word in line for line in segment(document)
        ])

    def textrank(sentences, func=jac_index):
        graph = networkx.Graph()
        graph.add_nodes_from(sentences)
        pairs = combinations(sentences, 2)
        for a, b in pairs:
            graph.add_edge(a, b, weight=func(a, b))
        page_rank = networkx.pagerank(graph, weight='weight')
        return {sentence: page_rank.get(sentence) for sentence in sentences}

    doc_by_tr = textrank(doc)
    wrd_by_tr = textrank(wrd, func=freq)

    li_doc = [(word, tr) for word, tr in doc_by_tr.items()]
    li_wrd = [(word, tr) for word, tr in wrd_by_tr.items()]

    sorted_doc = sorted(li_doc, key=lambda x: x[1], reverse=True)
    sorted_wrd = sorted(li_wrd, key=lambda x: x[1], reverse=True)

    return sorted_doc if unit=='sentence' else 'word'
