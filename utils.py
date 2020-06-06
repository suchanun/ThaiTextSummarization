import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re
from pythainlp.corpus import thai_stopwords
import pythainlp as pythai


def custom_preprocess(text):
    return re.sub(re.compile('({}|{})'.format("\b[+-]?\d+(?:\.\d+)?\b", "[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")), '',
                  pythai.util.normalize(text.lower()))


class ThaiRanker:
    def __init__(self, df, n_docs=1000, stoplist=thai_stopwords(),
                 smooth_idf=True, sentence_tokenize=pythai.tokenize.sent_tokenize,
                 word_tokenize=pythai.tokenize.word_tokenize, preprocessor=custom_preprocess):
        texts = df['body_text']
        self.sentence_tokenize = sentence_tokenize
        self.word_tokenize = word_tokenize
        self.n_docs = n_docs
        self.stoplist = stoplist
        self.smooth_idf = smooth_idf
        self.preprocessor = preprocessor

        count_vect = self.get_new_countvect()
        self.docs_word_freq = count_vect.fit_transform(texts).toarray()  # (docs, words)
        self.docs_word_freq = np.where(self.docs_word_freq > 0, 1, 0)
        self.docs_word_freq = np.sum(self.docs_word_freq, axis=0)
        # self.docs_word_freq /= np.sum(self.docs_word_freq)
        self.docs_vocab = count_vect.vocabulary_

    def get_vocabs(self):
        return self.vocab

    def get_new_countvect(self):
        return CountVectorizer(tokenizer=self.word_tokenize,
                               preprocessor=self.preprocessor, stop_words=self.stoplist)

    def process_text(self, text):
        ori_text_nodouble_newline = re.sub(r'\n+', '\n', text).strip()
        sentences = []
        start_indices_of_paragraphs = set()
        si = 0
        for paragraph in ori_text_nodouble_newline.split('\n'):
            start_indices_of_paragraphs.add(si)
            psentences = self.sentence_tokenize(paragraph)
            sentences.extend(psentences)
            n_psentences = len(psentences)
            si += n_psentences  # sentences in paragraph
            # self.sentence_tokenize(ori_text) self.sentence_tokenize(text)
        n_sentences = len(sentences)  # all sentences in text
        print('n_sentences == {}'.format(len(sentences)))

        text = text.lower()
        #         lemmatized_text = self.lemmatize(text)
        word_count_vectorizer = self.get_new_countvect()
        word_count = word_count_vectorizer.fit_transform([text]).toarray()[0]
        vocab = word_count_vectorizer.vocabulary_

        text_info = dict()
        text_info['word_freq'] = word_count / np.sum(word_count)
        text_info['vocab'] = vocab
        text_info['tfidf'] = self.get_tfidf_vect(text_info['word_freq'], text_info['vocab'])
        text_info['sentences'] = sentences
        text_info['n_sentences'] = n_sentences
        text_info['start_indices_of_paragraphs'] = start_indices_of_paragraphs
        return text_info

    def get_tfidf_vect(self, word_freq, vocab):
        d = np.log((1 + self.n_docs) / (1 + 0)) + 1
        tfidf = np.zeros(len(vocab))
        for word in vocab:
            idx = vocab[word]
            tf = word_freq[idx]
            #                         if re.search( r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''", word) is None:
            #                             tf = 0
            if word not in self.docs_vocab:
                idf = d
            else:
                idf = np.log((1 + self.n_docs) / (1 + self.docs_word_freq[self.docs_vocab[word]])) + 1
            tfidf[idx] = tf * idf

        return normalize([tfidf])[0]

    def get_score(self, sentence, text_info, words_already_in_summ, k, m, min_sentence_len):

        # remove numerical substrings and other non-alphabet substrings
        sentence = self.preprocessor(sentence)  # sentence.lower()

        tfidf_vect, vocab = text_info['tfidf'], text_info['vocab']
        words = [word for word in self.word_tokenize(sentence) if
                 (word not in self.stoplist)]
        n_words = 0
        score = 0
        for word in words:
            if word not in vocab:
                continue
            n_words += 1
            idx = vocab[word]
            tfidf = tfidf_vect[idx]
            if word in words_already_in_summ:
                tfidf *= k
            score += tfidf

        if n_words < min_sentence_len:
            if n_words == 0:
                return 0
            return (score / n_words) * m
        words_already_in_summ.update(words)
        return score / n_words

    # min_sentence_len > 0
    def rank_phrases(self, text, n, k=1, min_sentence_len=4, m=0.3):
        sentences_with_scores, start_indices_of_paragraphs = self.rank_sentences(text, k, min_sentence_len, m)
        groups, start_indices_of_paragraphs = self.group_sentences(sentences_with_scores, start_indices_of_paragraphs)
        paragraphs = self.sentence_groups_to_paragraph(groups, start_indices_of_paragraphs, n)
        return sentences_with_scores,paragraphs

    def rank_sentences(self, text, k=0.5, min_sentence_len=4, m=0.3):
        text_info = self.process_text(text)
        sentences = [(i, sentence) for i, sentence in enumerate(text_info['sentences'])]
        selected_sentences = []
        words_in_summ = set()
        n_selected = 0
        while n_selected < text_info['n_sentences']:
            candidate_sentences = [
                (i, sentence, self.get_score(sentence, text_info, words_in_summ, k, m, min_sentence_len)) for
                i, sentence in sentences]
            i, selected_sentence, score = max(candidate_sentences, key=lambda tup: tup[2])
            selected_sentences.append((i, selected_sentence, score))
            sentences.remove((i, selected_sentence))
            words_in_summ.update(self.word_tokenize(selected_sentence))
            n_selected += 1
        return selected_sentences, text_info['start_indices_of_paragraphs']

    #     def rank_and_group_sentences(self, text, k=1, min_sentence_len=4, m=0.3):
    #         selected_sentences,start_indices_of_paragraphs = self.rank_sentences(text, k, min_sentence_len, m)
    #         return self.group_sentences(selected_sentences,start_indices_of_paragraphs)
    def find_paragraph_index(self, start_indices_of_paragraphs, sentence_index):
        last_p_index = 0
        for current_p_index in start_indices_of_paragraphs:
            if sentence_index == current_p_index:
                return current_p_index
            elif sentence_index < current_p_index:
                return last_p_index
            last_p_index = current_p_index
        return last_p_index

    def group_sentences(self, sentences_with_scores, start_indices_of_paragraphs):
        ans = []
        if not sentences_with_scores:
            return ans
        sentences_by_order_in_text = sorted(sentences_with_scores, key=lambda tup: tup[0])
        sorted_paragraphs_indices = sorted(list(start_indices_of_paragraphs))
        irank = 0
        sentence_order = sentences_with_scores[irank][0]
        max_n = len(sentences_by_order_in_text)
        selected_count = 0
        selected = set()
        groups = []
        while selected_count < max_n:
            group = []
            while sentence_order in selected:
                irank += 1
                sentence_order = sentences_with_scores[irank][0]

            # print(sentences_with_scores)

            score = sentences_by_order_in_text[sentence_order][2]
            group_paragraph = self.find_paragraph_index(sorted_paragraphs_indices, sentence_order)

            # left side
            left_order = sentence_order - 1
            while left_order >= 0 and sentences_by_order_in_text[left_order][2] <= score \
                    and left_order not in selected and self.find_paragraph_index(sorted_paragraphs_indices,
                                                                                 left_order) == group_paragraph:
                score = sentences_by_order_in_text[left_order][2]
                left_order -= 1
            for order in range(left_order + 1, sentence_order + 1):
                group.append(sentences_by_order_in_text[order])
                selected.add(order)
                selected_count += 1
            # right side
            score = sentences_by_order_in_text[sentence_order][2]
            right_order = sentence_order + 1
            while right_order < max_n and sentences_by_order_in_text[right_order][2] <= score \
                    and right_order not in selected and self.find_paragraph_index(sorted_paragraphs_indices,
                                                                                  right_order) == group_paragraph:
                score = sentences_by_order_in_text[right_order][2]
                right_order += 1
            for order in range(sentence_order + 1, right_order):
                group.append(sentences_by_order_in_text[order])
                selected.add(order)
                selected_count += 1
            groups.append(group)
        return groups, start_indices_of_paragraphs
        # while right_ci<

    def sentence_groups_to_paragraph(self, sentence_groups, start_indices_of_paragraphs, n_groups):
        paragraphs = dict()
        for p_index in start_indices_of_paragraphs:
            paragraphs[p_index] = []
        start_indices_of_paragraphs = sorted(list(start_indices_of_paragraphs))

        for group in sentence_groups[:n_groups]:
            sentence = group[0]
            p_index = self.find_paragraph_index(start_indices_of_paragraphs, sentence[0])  # ][0])
            paragraphs[p_index].append(group)
        for p_index in start_indices_of_paragraphs:
            paragraphs[p_index] = sorted(paragraphs[p_index], key=lambda tup: tup[0])
        return paragraphs



