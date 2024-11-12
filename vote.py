import openai
from openai import OpenAI
import yaml
from typing import Dict, Optional, List, Tuple

client = None
with open("./key-openai.yaml", 'r') as f:
    api_keys = yaml.safe_load(f)

MODEL_CONFIGS = [
    {"model": "openai/gpt-4o", "temperature": 0.5, "max_tokens": 15000},
    {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.5, "max_tokens": 15000},
    {"model": "deepseek/deepseek-chat", "temperature": 0.5, "max_tokens": 15000},
    {"model": "qwen/qwen-2.5-72b-instruct", "temperature": 0.5, "max_tokens": 15000},
    {"model": "01-ai/yi-large", "temperature": 0.5, "max_tokens": 15000},
]

with open("./vote-sys-inst.md", 'r', encoding='utf-8') as f:
    VOTE_SYS_INST = f.read()


def chat(**kwargs) -> str:
    kwargs["stream"] = False
    return client.chat.completions.create(**kwargs).choices[0].message.content


def parse_score(response) -> Optional[List[float]]:
    details = response.split("## ")
    details = [detail.strip() for detail in details]
    # remove empty strings
    details = list(filter(None, details))

    if (len(details) != 15):
        print(f"Expected 15 details, got {len(details)}")
        return None

    titles = [
        "1. 合理性",
        "2. 新颖程度",
        "3. 悬念",
        "4. 反转和惊喜",
        "5. 期待感",
        "6. 目标",
        "7. 读者偏好",
        "8. 设定复杂性",
        "9. 情节复杂性",
        "10. 代入感",
        "11. 情感波动",
        "12. 一致性",
        "13. 相关度",
        "14. 结局",
        "15. 情节分配"
    ]

    # check if the titles are in the correct order
    for i in range(len(titles)):
        if titles[i] not in details[i]:
            print(f"Title {titles[i]} not found in details[{i}]")
            return None

    ret = []
    # parse the score
    for i in range(len(details)):
        lines = details[i].split("\n")
        lines = [line.strip() for line in lines]
        for line in lines:
            if line.startswith("score:"):
                score = line.split(":")[-1].strip()
                try:
                    new_score = float(score)
                    ret.append(new_score)
                except ValueError:
                    print(f"Failed to parse score: {score} for title: {titles[i]}")
                    return None

    return ret


def vote_one_model_openai(outline: str, model_config) -> Dict[str, Tuple[List[float], str]]:
    global client 
    client = OpenAI(api_key=api_keys["key"], base_url="https://api.openai.com/v1")
    
    MAX_RETRY = 3
    messages = [
        {"role": "system", "content": VOTE_SYS_INST},
        {"role": "user", "content": outline}
    ]
    model_config["messages"] = messages

    try_cnt = 0
    scores = None
    while try_cnt < MAX_RETRY:
        response = chat(**model_config)
        scores = parse_score(response)
        if scores is not None:
            break
        else:
            print(f"Failed to parse scores for model: {model_config['model']}")
            print(response)
        try_cnt += 1

    if scores is None:
        raise Exception("Failed to parse scores")

    return {model_config["model"]: (scores, response)} #返回一个字典，key是模型名，value是一个元组，元组的第一个元素是分数列表，第二个元素是response


def vote_one_model(outline: str, model_config) -> Dict[str, Tuple[List[float], str]]:
    global client
    client = OpenAI(api_key=api_keys["key_openrouter"], base_url="https://openrouter.ai/api/v1")
    
    MAX_RETRY = 3
    messages = [
        {"role": "system", "content": VOTE_SYS_INST},
        {"role": "user", "content": outline}
    ]
    model_config["messages"] = messages

    try_cnt = 0
    scores = None
    while try_cnt < MAX_RETRY:
        response = chat(**model_config)
        scores = parse_score(response)
        if scores is not None:
            break
        else:
            print(f"Failed to parse scores for model: {model_config['model']}")
            print(response)
        try_cnt += 1

    if scores is None:
        raise Exception("Failed to parse scores")

    return {model_config["model"]: (scores, response)} #返回一个字典，key是模型名，value是一个元组，元组的第一个元素是分数列表，第二个元素是response


def vote_all(outline: str) -> Dict[str, List[float]]:

    ret = {}
    for model in MODEL_CONFIGS:
        try:
            scores = vote_one_model(outline, model)
            ret.update(scores)
        except Exception as e:
            print(f"Failed to vote for model: {model['model']}")
            print(e)

    # try:
    #     scores = vote_one_model(outline, {"model": "deepseek/deepseek-chat", "temperature": 0.5, "max_tokens": 15000})
    #     ret.update(scores)
    # except Exception as e:
    #     print(f"Failed to vote for model: {model['model']}")
    #     print(e) 

    return ret
