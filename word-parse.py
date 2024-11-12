import numpy as np
from typing import TypedDict, List
import yaml


class OptionalWords(TypedDict):
    weight: float
    words: List[str]
    max_num: int # 随机抽取的最大数量
    min_num: int # 随机抽取的最小数量

class GeneralWords(TypedDict): # 大分类
    weight: float
    fixed_words: str
    optional_words: OptionalWords

def parse_word_list(txt) -> OptionalWords:
    ret = OptionalWords()
    ret['words'] = txt.split(' ') # TODO: support random sample size
    ret['words'] = remove_diplicate_empty(ret['words'])
    return ret

def remove_diplicate_empty(words: List[str]) -> List[str]:
    ret = [word.strip() for word in words if word.strip() != ""]
    return list(set(ret))

def parse_param(word_list): 
    ret_dict = {}
    i = 0
    while i < len(word_list):
        word = word_list[i]
        if word.startswith("-"): # example: -w 0.1 -r 1 -x 1
            ret_dict[word[1:]] = word_list[i+1]
            i += 1
        i += 1
    return ret_dict
            

def parse_general_words(obj) -> List[GeneralWords]:
    general_words = []
    for item in obj:
        new_item = GeneralWords()
        new_item['weight'] = 1 if "w" not in item else float(item['w'])
        new_item['fixed_words'] = item['m'].strip()
        if "o" in item:
            if type(item['o']) == str:
                new_item['optional_words'] = [parse_word_list(item['o'])]
            else:
                new_item['optional_words'] = []
                for txt in item['o']:
                    new_item['optional_words'].append(parse_word_list(txt))
        general_words.append(new_item)
    return general_words


with open('./words/general.yaml', 'r', encoding='utf-8') as f:
    general = yaml.load(f, Loader=yaml.FullLoader)

general_cates = parse_general_words(general)

import word_sampler as ws
print(ws.sample_general(general_cates))

import json

