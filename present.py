from utils import *
import streamlit as st
import pickle
import pythainlp as thainlp
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from visualizer import *

@st.cache(allow_output_mutation=True)
def load_data():
    return pickle.load(open('data/prachathai1000.pkl','rb')), pickle.load(open('data/ranker_v2','rb'))


def run_app():
    df,ranker = load_data()
    n_groups = st.sidebar.number_input('number of sentences in summary', value=5, min_value=1)
    article_number = st.sidebar.number_input('article number', value=0, min_value=0, max_value=999)
    st.sidebar.markdown('or')
    random = st.sidebar.button('random an article')
    if random:
        doc_i = np.random.randint(1000)
    else:
        doc_i = article_number
    st.header('Prachathai {}'.format(doc_i))
    item = df.iloc[doc_i]
    text = item['body_text']
    sentences_with_scores, paragraphs = ranker.rank_phrases(text, n_groups)
    info = dict()
    info['title'] = item['title']
    info['text'] = text
    info['paragraphs'] = paragraphs
    info['sentences_with_scores'] = sentences_with_scores
    Displayer.show_summary_and_text(info)

    # st.header(df.iloc[doc_i]['title'])
    # st.markdown(text.replace('\n','\n\n'))
    # info = dict()
    # info['text'] = text

    #info['my_model_result'] = result
    #by_order_in_text = sorted(result, key=lambda tup:tup[0])
    #scores_by_order_in_text = np.array([tup[2] for tup in by_order_in_text ])
    #peaks, _ = find_peaks(scores_by_order_in_text , height=0, distance=1)

    # fig = plt.figure()
    # plt.plot(scores_by_order_in_text,'.-')
    # plt.plot(peaks, scores_by_order_in_text[peaks], 'x')
    # st.write(fig)
    # Displayer.display_figure(peaks, scores_by_order_in_text[peaks],title='score by pos', xaxis_title='pos', yaxis_title='score', mode='lines+markers')
    # Displayer.display_figure(x=list(range(len(result))), y=scores_by_order_in_text, title='score by pos', xaxis_title='pos', yaxis_title='score', mode='lines+markers')

    # Displayer.show_custom(info,n_sentences)
    #
    # item = df.iloc[doc_i]
    # paragraphs = item['body_text'].split('\n')
    # st.header(item['title'])
    # i = 0
    # for paragraph in paragraphs:
    #     # st.markdown('\t{}'.format(paragraph))
    #     sentences = thainlp.sent_tokenize(paragraph)
    #     for sentence in sentences:
    #         st.markdown('{}: {}'.format(i,sentence))
    #         i+=1
    #         words = thainlp.word_tokenize(sentence)
    #         # st.markdown('words: {}'.format(' '.join(words)))
    #         st.markdown(thainlp.tag.pos_tag(words))
#,corpus='orchid_ud'
if __name__ == "__main__":
    run_app()