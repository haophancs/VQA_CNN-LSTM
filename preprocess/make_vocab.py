import json
import os
import re
from collections import defaultdict

src_dir = "/home/mlworker/Khiem/beit3/datasets/viclevr"
saving_dir = "./data/viclevr/preprocessed/vocab"

top_answer = 1000

def make_q_vocab():

    dataset = os.listdir(src_dir + '/vqa')
    regex = re.compile(r'(\W+)')
    q_vocab = []
    for file in dataset:
        if 'questions.json' not in file:
            continue
        path = os.path.join(src_dir, 'vqa', file)
        with open(path, 'r') as f:
            q_data = json.load(f)
        question = q_data['questions']
        for idx, quest in enumerate(question):

            split = regex.split(quest['question'].lower())
            tmp = [w.strip() for w in split if len(w.strip()) > 0]
            q_vocab.extend(tmp)

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab.insert(0, '<pad>')
    q_vocab.insert(1, '<unk>')

    if not os.path.exists(saving_dir): os.makedirs(saving_dir)
    with open(saving_dir + '/question_vocabs.txt', 'w') as f:
        f.writelines([v+'\n' for v in q_vocab])

    print(f"total word:{len(q_vocab)}")

def make_a_vocab(top_answer):

    answers = defaultdict(lambda :0)
    dataset = os.listdir(src_dir + '/vqa')
    for file in dataset:
        if 'annotations.json' not in file:
            continue
        path = os.path.join(src_dir, 'vqa', file)
        with open(path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        for label in annotations:
            for ans in label['answers']:
                vocab = ans['multiple_choice_answer']
                if re.search(r'[^\w\s]', vocab):
                    continue
                answers[vocab] += 1

    answers = sorted(answers, key=answers.get, reverse= True) # sort by numbers
    top_answers = ['<unk>'] + answers[:top_answer-1]
    with open(saving_dir + '/annotation_vocabs.txt', 'w') as f :
        f.writelines([ans+'\n' for ans in top_answers])

    print(f'The number of total words of answers: {len(answers)}')
    print(f'Keep top {top_answers} answers into vocab' )

if __name__ == "__main__":

    make_q_vocab()
    make_a_vocab(top_answer)