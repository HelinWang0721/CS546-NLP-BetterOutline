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
    words = txt.split(" ")
    ret['words'] = txt.split(' ')
    i = 0
    while i < len(words):
        if words[i].startswith("-"): # config option
            option = words[i]
            # remove this from words
            words.pop(i)
            i -= 1
            if option.startswith("-r"): # random min and max
                payload = option[2:].split(",")
                ret['min_num'] = int(payload[0])
                ret['max_num'] = int(payload[1])
            elif option.startswith("-w"): # weight
                ret['weight'] = float(option[2:])
            else:
                print("Unknown option: ", option)
        i += 1

    ret['words'] = remove_diplicate_empty(words)
    return ret

def remove_diplicate_empty(words: List[str]) -> List[str]:
    ret = [word.strip() for word in words if word.strip() != ""]
    return list(set(ret))
            

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
