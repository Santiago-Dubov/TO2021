import numpy as np
import nltk
nltk.download('punkt')
from scipy import spatial
import pipelines
import random


def create_questions(input_text_file):
    # text file should contain summaries of each page seperated by a line
    nlp = pipelines.pipeline("question-generation", model="valhalla/t5-base-qg-hl")
    with open(input_text_file) as f:
        raw = f.read()
    raw_sep = raw.split('\n')

    for summary in raw_sep:
        question_answer = nlp(text)
    embeddings_dict = {}

    # creating the embeddings dictionary to get word vectors 
    with open("glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    def find_closest_embeddings(embedding):
        return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
    
    # creating our multiple choice questions 
    for pair in question_answer:
        answer = pair['answer']
        # First handle case of answer being a number 
        try:
            n = float(answer)
            pair['alternatives'] = []
            if n!= 0:
                pair['alternatives'].append('0')
            else:
                pair['alternatives'].append('1')
            pair['alternatives'].append(str(10*n))
            pair['alternatives'].append(str(int(n*random.random())))
            continue
        except ValueError:
            pass
        answer = answer.split()
        if len(answer) == 1:
            try:
                options = find_closest_embeddings(embeddings_dict[answer[0]])[8:15]
                #print(options)
                pair['alternatives'] = random.choices(options, k=3)
            except:
                continue
        elif len(answer) == 2:
            try:
                options = find_closest_embeddings(embeddings_dict[answer[0].lower()]+embeddings_dict[answer[1].lower()])[8:15]
                pair['alternatives'] = random.choices(options, k=3)
            except:
                pass
        else:
            continue
    return question_answer


if __name__ == '__main__':
    sample_text_file = 'biology_notes.txt'
    create_questions(sample_text_file)