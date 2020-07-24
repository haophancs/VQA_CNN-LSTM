import os
import re
import json
import glob
import numpy as np

image_dir = "../data/resize_image"
annotation_dir = "/HDD-1_data/dataset/VQA-v2/Annotations"
question_dir = "/HDD-1_data/dataset/VQA-v2/Questions"
output_dir = "../data/"

def preprocessing(question, annotation_dir, image_dir, labeled):

    with open(question, 'r') as f:
        data = json.load(f)
        questions = data['questions']
        data_type = data['data_subtype']

    if labeled:
        template = annotation_dir + f'/*{data_type}*.json'
        annotation_path = glob.glob(template)[0]
        with open(annotation_path) as f:
            annotations = json.load(f)['annotations']
        question_dict = {ans['question_id']: ans for ans in annotations}

    match_top_ans.unk_ans = 0
    dataset = [None]*len(questions)
    for idx, qu in enumerate(questions):
        if (idx+1) % 10000 == 0:
            print(f'processing {data_type} data: {idx+1}/{len(questions)}')
        qu_id = qu['question_id']
        qu_sentence = qu['question']
        qu_tokens = tokenizer(qu_sentence)
        img_id = qu['image_id']
        img_name = 'COCO_' + data_type + '_{:0>12d}.jpg'.format(img_id)
        img_path = os.path.join(image_dir, data_type, img_name)

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

    np.save(output_dir + f'{data_type[:-4]}.npy', np.array(dataset))

    print(f'total {match_top_ans.unk_ans} out of {len(questions)} answers are <unk>')
    print('ok')

def tokenizer(sentence):

    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def match_top_ans(annotation_ans):

    annotation_dir = output_dir + 'annotation_vocabs.txt'
    if "top_ans" not in match_top_ans.__dict__:
        with open(annotation_dir, 'r') as f:
            match_top_ans.top_ans = {line.strip() for line in f}
    annotation_ans = {ans['answer'] for ans in annotation_ans}
    valid_ans = match_top_ans.top_ans & annotation_ans

    if len(valid_ans) == 0:
        valid_ans = ['<unk>']
        match_top_ans.unk_ans += 1

    return annotation_ans, valid_ans


if __name__ == "__main__":

    for file in os.listdir(question_dir):

        labeled = False if "test" in file else True
        question = os.path.join(question_dir, file)
        preprocessing(question, annotation_dir, image_dir, labeled)
