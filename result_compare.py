from ctypes import sizeof
import json
from pickletools import read_uint1
import re

from cv2 import compare

import vote

with open('./Fine_tune/openai_FT_score.json', encoding='utf-8') as f:
    opai_score1 = json.load(f)

with open('./Fine_tune/openai_BS_score.json', encoding='utf-8') as f:
    opai_score2 = json.load(f)

openai_score1_list = []
openai_score2_list = []


for entry in opai_score1:
    if "gpt-4o" in entry:
        scores = entry["gpt-4o"][0]
        openai_score1_list.append(scores) 

for entry in opai_score2:
    if "gpt-4o" in entry:
        scores = entry["gpt-4o"][0] 
        openai_score2_list.append(scores) 


total_openai_score1_list = []
total_openai_score2_list = []

for scores in openai_score1_list:
    total_openai_score1_list.append(sum(scores))

for scores in openai_score2_list:
    total_openai_score2_list.append(sum(scores))

# The following code is use for other models

with open('./Fine_tune/FT_allmodel_score.json', encoding='utf-8') as f:
    outline1_scores = json.load(f)

with open('./Fine_tune/BS_allmodel_score.json', encoding='utf-8') as f:
    outline2_scores = json.load(f)

multi_finetune_scores = {}
multi_original_scores = {}


multi_finetune_scores["gpt-4o"] = total_openai_score1_list
multi_original_scores["gpt-4o"] = total_openai_score2_list

for element in outline1_scores:
    for k, v in element.items():
            if k not in multi_finetune_scores:
                multi_finetune_scores[k] = []
            multi_finetune_scores[k].append(sum(v[0]))

for element in outline2_scores:
    for k, v in element.items():
            if k not in multi_original_scores:
                multi_original_scores[k] = []
            multi_original_scores[k].append(sum(v[0]))
   
# print(multi_finetune_scores)
def vote_result(multi_finetune_scores, multi_original_scores):
    vote_results = {"finetune": [], "original": []}
    i = 0
    while True:
        in_range_flag = False
        count_finetune_win = 0
        count_original_win = 0
        for k,v in multi_finetune_scores.items():
            if not ("4o" in k):# or "claude" in k or "qwen" in k):
                continue
            if i < len(v):
                in_range_flag = True
            else:
                continue
            original_score = multi_original_scores[k][i]
            finetune_score = multi_finetune_scores[k][i]
            
            if finetune_score > original_score:
                count_finetune_win += 1
            else:
                count_original_win += 1
        
        if not in_range_flag:
            break
        if count_finetune_win > count_original_win:
            vote_results['finetune'].append(i)
        else:
            vote_results['original'].append(i)
        i += 1
    return vote_results

# Compute vote results
vote_results = vote_result(multi_finetune_scores, multi_original_scores)

print(vote_results)

# Final decision
if len(vote_results['finetune']) > len(vote_results['original']):
    print("\nOverall Result: After Fine_tune is better")
else:
    print("\nOverall Result: Before Fine_tune is better")