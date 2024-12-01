from ctypes import sizeof
import json
from pickletools import read_uint1
import re

from cv2 import compare

import vote

with open('./openai_outline1_score.json', encoding='utf-8') as f:
    opai_score1 = json.load(f)

with open('./openai_outline2_score.json', encoding='utf-8') as f:
    opai_score2 = json.load(f)

score1 = opai_score1["gpt-4o"][0]
score2 = opai_score2["gpt-4o"][0]

total_score1 = sum(score1)
total_score2 = sum(score2)


with open('./outline1_allmodel_score.json', encoding='utf-8') as f:
    outline1_scores = json.load(f)

with open('./outline2_allmodel_score.json', encoding='utf-8') as f:
    outline2_scores = json.load(f)

for k, v in opai_score1.items():
    outline1_scores[0][k] = v

for k, v in opai_score2.items():
    outline2_scores[0][k] = v

def vote_result(scores1, scores2):
    result = {'Outline1':[],'Outline2':[]}

    for model_name in scores1[0]:
        print(model_name)
        # ensure both score have the score from this model
        if model_name not in scores2[0]:
            continue

        score1_sum = sum(scores1[0][model_name][0])
        score2_sum = sum(scores2[0][model_name][0])
        print(f"score1 sum {score1_sum}, score2 sum {score2_sum}")

        if score1_sum > score2_sum:
            result['Outline1'].append(scores1[0][model_name][0])
            
        else:
            result['Outline2'].append(scores2[0][model_name][0])

    return result

vote_results = vote_result(outline1_scores, outline2_scores)

if len(vote_results['Outline1']) > len(vote_results['Outline2']):
    print("Outline1 is better")
else:
    print("Outline2 is better")