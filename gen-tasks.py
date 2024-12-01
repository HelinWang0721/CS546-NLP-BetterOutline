from word_sampler import sample_optional, sample_general
from word_parse import parse_general_words, parse_word_list
import json
import numpy as np
import yaml

with open('./words/general.yaml', 'r', encoding='utf-8') as f:
    general = yaml.load(f, Loader=yaml.FullLoader)

general_cates = parse_general_words(general) # general categories

optional_words = []
with open('./words/other_words.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if line.strip() == "":
            continue
        optional_words.append(parse_word_list(line.strip()))

def gen_tasks(num):
    tasks = []
    tasks_str_dict = {}
    i = 0
    while i < num:
        task = {}
        task['m_words'] = sample_general(general_cates)
        task['o_words'] = sample_optional(20, optional_words)
        g_list = task['m_words'].split(' ')
        o_list = task['o_words'].split(' ')
        set_list = list(set(g_list + o_list))
        # sort the str
        set_list.sort()
        task_str = ' '.join(set_list)
        if task_str in tasks_str_dict:
            continue
        tasks_str_dict[task_str] = 1
        task["id"] = i
        tasks.append(task)
        i += 1
    return tasks


tasks = gen_tasks(4000)
with open('./tasks.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, ensure_ascii=False, indent=4)