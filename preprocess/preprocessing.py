import os
import re
import json
import glob
import numpy as np
from tqdm import tqdm

image_dir = "./data/viclevr/preprocessed/resized_images"
annotation_dir = "/home/mlworker/Khiem/beit3/datasets/viclevr/vqa"
question_dir = "/home/mlworker/Khiem/beit3/datasets/viclevr/vqa"
output_dir = "./data/viclevr/preprocessed"

def preprocessing(question, annotation_dir, image_dir, labeled):

    with open(question, 'r') as f:
        data = json.load(f)
        questions = data['questions']
        subset = data['data_subtype']

    if labeled:
        template = os.path.join(annotation_dir, f'*{subset}_annotations.json')
        annotation_path = glob.glob(template)[0]
        with open(annotation_path) as f:
            annotations = json.load(f)['annotations']
        question_dict = {ans['question_id']: ans for ans in annotations}

    match_top_ans.unk_ans = 0
    dataset = [None]*len(questions)
    for idx, qu in tqdm(enumerate(questions)):
        qu_id = qu['question_id']
        qu_sentence = qu['question']
        qu_tokens = tokenizer(qu_sentence)
        img_id = qu['image_id']
        img_name = 'vi_clevr_' + subset + '_{:0>6d}.png'.format(img_id)
        img_path = os.path.join(image_dir, subset, img_name)

        info = {'img_name': img_name,
                'img_path': img_path,
                'qu_sentence': qu_sentence,
                'qu_tokens': qu_tokens,
                'qu_id': qu_id}

        if labeled:
            annotation_ans = question_dict[qu_id]['answers']
            all_ans, valid_ans = match_top_ans(annotation_ans)
            info['all_ans'] = all_ans
            info['valid_ans'] = valid_ans

        dataset[idx] = info

    print(f'total {match_top_ans.unk_ans} out of {len(questions)} answers are <unk>')
    return dataset

def tokenizer(sentence):

    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def match_top_ans(annotation_ans):

    annotation_dir = os.path.join(output_dir + 'annotation_vocabs.txt')
    if "top_ans" not in match_top_ans.__dict__:
        with open(annotation_dir, 'r') as f:
            match_top_ans.top_ans = {line.strip() for line in f}
    annotation_ans = {ans['answer'] for ans in annotation_ans}
    valid_ans = match_top_ans.top_ans & annotation_ans

    if len(valid_ans) == 0:
        valid_ans = ['<unk>']
        match_top_ans.unk_ans += 1

    return annotation_ans, valid_ans

def main():

    processed_data = {}
    for file in os.listdir(question_dir):
        if 'questions.json' not in file:
            continue
        datatype = 'train' if 'train' in file else ('val' if val in file else 'test')
        question = os.path.join(question_dir, file)
        processed_data[datatype] = preprocessing(question, annotation_dir, image_dir, labeled=True)

    processed_data['train-val'] = processed_data['train'] + processed_data['val']
    for key, value in processed_data.items():
        np.save(os.path.join(output_dir, f'{key}.npy'), np.array(value))

if __name__ == "__main__":

    main()