import plotly.graph_objects as go
import streamlit as st
import plotly
import numpy as np
class Displayer:
    @staticmethod
    def show_summary_and_text(info):
        st.header(info['title'])
        text = info['text']
        paragraphs = info['paragraphs']
        sentences_with_scores = sorted(info['sentences_with_scores'],key=lambda tup: tup[0])
        paragraph_numbers = sorted(list(paragraphs.keys()))
        st.subheader('My Summary')
        last_phrase_index = -1
        prev_phrases = ''
        selected_sentences = []
        selected_indices = []
        print(paragraphs)
        for p_num in paragraph_numbers:
            paragraph = paragraphs[p_num]
            # st.text('p_num == {}'.format(p_num))
            for phrases_group in paragraph:
                sentences_tokens = [tup[1] for tup in phrases_group]
                selected_indices.extend([tup[0] for tup in phrases_group])
                sentences = ' '.join(sentences_tokens)
                selected_sentences.extend(sentences_tokens)
                phrase_index = phrases_group[0][0]
                # st.text('phrase_index == {}, sentences == '.format(phrase_index, sentences))
                #print(repr(sentences))
                # print('last_phrase_index == {}'.format(last_phrase_index))
                if phrase_index == last_phrase_index + 1:
                    prev_phrases += ' '+ sentences
                else:
                    if prev_phrases != '':# start new phrases
                        st.markdown(prev_phrases)
                    prev_phrases = sentences
                last_phrase_index = phrases_group[-1][0]

        if prev_phrases != '':
            st.markdown(prev_phrases)
        sentences_group = dict()
        sentences_group[0] = selected_sentences
        highlighted_text = Highlighter.get_highlighted_html(text, sentences_group).replace('\n','\n\n')

        y = [tup[2] for tup in sentences_with_scores]
        Displayer.display_selected_in_graph(x=list(range(len(sentences_with_scores))),y=y,selected_indices=selected_indices)
        st.subheader('Highlights Displayed')
        st.markdown(highlighted_text, unsafe_allow_html=True)
        print(repr(highlighted_text))
        # for tup in sentences_with_scores:
        #     st.markdown(tup)
        # st.markdown(text)


    @staticmethod
    def display_selected_in_graph(x,y,selected_indices,mode='lines+markers'):#color_selected='tomato',
        # n = len(selected_indices)
        # y_not_selected = [y[i] for i in range(n) if i not in selected_indices]
        # y_selected = [y[i] for i in selected_indices]
        marker_color = np.zeros(len(x))#['tomato']
        # for idx in selected_indices:
        #     marker_color[idx]='ocean'
        np.put(marker_color,selected_indices,1)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode,marker=dict(color=marker_color,\
                                                            colorscale=plotly.colors.sequential.Bluered)))#,colorscale='Hot'
        st.plotly_chart(fig)
    @staticmethod
    def display_figure(x,y,title,xaxis_title,yaxis_title,mode='lines+markers'):

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode ))
        fig.update_layout(title=title,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        st.plotly_chart(fig)

class Highlighter:

    @staticmethod
    def get_highlighted_html(text,sentences_group, intersect_group_id=2):
        highlighted_indices = Highlighter.get_highlight_indices(text, sentences_group, intersect_group_id)
        return Highlighter.compute_highlighted_text(text, highlighted_indices)

    @staticmethod
    def get_highlight_indices( text, sentences_group, intersect_group_id):
        def has_duplicate( start_end_indices, start_idx):
            for index in start_end_indices:
                if index[0] == start_idx:
                    return True
            return False

        text = text.lower()
        indices_with_group_id = []
        indices_by_groupID = dict()
        for group_id in sentences_group:
            indices_by_groupID[group_id] = set()
            for sentence in sentences_group[group_id]:
                sentence = sentence.lower().strip()
                start_idx = text.find(sentence)
                while has_duplicate(indices_by_groupID[group_id], start_idx):
                    start_idx = text.find(sentence, start_idx + len(sentence))
                if start_idx == -1:
                    continue
                index = (start_idx, start_idx + len(sentence))
                indices_by_groupID[group_id].add(index)
        intersect_indices = set.intersection(*[indices_by_groupID[gid] for gid in sentences_group])
        for group_id in sentences_group:
            indices = indices_by_groupID[group_id]
            for index in indices:
                if index not in intersect_indices:
                    indices_with_group_id.append(index + (group_id,))
        for idx in intersect_indices:
            indices_with_group_id.append(idx + (intersect_group_id,))
        ans =  sorted(indices_with_group_id, key=lambda my_tuple: my_tuple[0])

        return ans

    @staticmethod
    def compute_highlighted_text(text, indices, colors={0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}):
        highlighted_text = ''
        last_pos = 0

        for index in indices:
            start, end, color_id = index
            color_code = colors[color_id]

            highlighted_text += text[last_pos:start] + '<span style="background-color: {}">'.format(color_code) + text[
                                                                                                                  start:end] + '</span>'
            last_pos = end
        highlighted_text += text[last_pos:]
        return highlighted_text