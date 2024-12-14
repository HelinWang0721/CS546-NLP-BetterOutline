import openai
import numpy as np
from typing import List
from string import Template
import yaml
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-4o") # to count tokens, making price estimation easier

with open("./gen-prompt.md", 'r', encoding="utf-8") as f:
    prompt1 = f.read()
    prompt1 = prompt1.split("<EOF>")
    prompt2 = prompt1[1].strip() + "\n"
    prompt1 = prompt1[0].strip() + "\n"


def get_rand_n(min_n, max_n, percentage): # get a random number from normal distribution
    mu = percentage * (max_n - min_n) + min_n # mean of the distribution
    sigma = (max_n - min_n) / 6 # standard deviation of the distribution
    n = np.random.normal(mu, sigma) # a random number from normal distribution
    n = np.clip(n, min_n, max_n)
    return int(n)


RANGE_CHARACTERS = [3, 6] # inclusive
RANGE_SCENES = [4, 10]
RANGE_EVENTS = [8, 18]
def get_prompt(mandatory_words: str, optional_words: str):
    rand_percentage = np.random.rand() # uniform distribution
    n_characters = get_rand_n(RANGE_CHARACTERS[0], RANGE_CHARACTERS[1], rand_percentage) 
    n_scenes = get_rand_n(RANGE_SCENES[0], RANGE_SCENES[1], rand_percentage)
    n_events = get_rand_n(RANGE_EVENTS[0], RANGE_EVENTS[1], rand_percentage)
    
    global prompt1, prompt2
    prompt = Template(prompt1)
    prompt = prompt.substitute(mandatory_words=mandatory_words, 
                               optional_words=optional_words, 
                                n_characters=n_characters,
                                n_scenes=n_scenes,
                                n_events=n_events)
    
    return [prompt, prompt2]



with open("./key-openai.yaml", 'r') as f:
    api_keys = yaml.safe_load(f)
client = openai.OpenAI(
    api_key=api_keys['key']
)

def count_tokens(text: str):
    return len(tokenizer.encode(text))

def chat(**kwargs):
    kwargs["stream"] = False
    all_prompt = ""
    for message in kwargs["messages"]:
        all_prompt += message["content"] + "\n"
    print(f"prompt tokens: {count_tokens(all_prompt)}")
    ret = client.chat.completions.create(**kwargs).choices[0].message.content
    print(f"response tokens: {count_tokens(ret)}")
    return ret


def check_outline_completeness(outline):
    title_list = ["## 故事氛围", "## 故事背景", 
                  "## 人设", "## 场景", "## 目的", "## 高潮和结局", 
                  "## 事件大纲"]
    for title in title_list:
        if title not in outline:
            return False
    return True


# example:
# mandatory_words = "古代, 现代, 时空冲突, 化敌为友, 冷兵器, 感动"
# optional_words = "冒失, 教授, 细眉 ..." 至少 12-15 个词

def generate_outline(mandatory_words: str, optional_words: str):
    prompt = get_prompt(mandatory_words, optional_words)
    messages = [
        {"role": "user", "content": prompt[0]}
    ]
    try_count = 0
    response_list = []
    while try_count < 3:
        response = chat(
            model="o1-preview", 
            messages=messages
            # temperature=0.75,
            # max_completion_tokens=8000
        )
        if check_outline_completeness(response):
            messages.append({"role": "assistant", "content": response})
            response_list.append(response)
            break
        if try_count == 2:
            print(f"Failed to generate outline")
            print(f"mandatory words: {mandatory_words}")
            print(f"optional words: {optional_words}")
            print("response:")
            print(response)
            raise Exception("Failed to generate outline")
        try_count += 1

    messages.append({"role": "user", "content": prompt[1]})
    response = chat(
        model="gpt-4o", 
        messages=messages,
        temperature=0.75,
        # max_tokens=15000
    )
    messages.append({"role": "assistant", "content": response})
    response_list.append(response)
    
    return prompt, messages, response_list


def generate_BS_outline(mandatory_words: str, optional_words: str):
    prompt = get_prompt(mandatory_words, optional_words)
    messages = [
        {"role": "user", "content": prompt[0]}
    ]
    try_count = 0
    response_list = []
    while try_count < 3:
        response = chat(
            model="gpt-4o-2024-08-06", 
            messages=messages
            # temperature=0.75,
            # max_completion_tokens=8000
        )
        if check_outline_completeness(response):
            messages.append({"role": "assistant", "content": response})
            response_list.append(response)
            break
        if try_count == 2:
            print(f"Failed to generate outline")
            print(f"mandatory words: {mandatory_words}")
            print(f"optional words: {optional_words}")
            print("response:")
            print(response)
            raise Exception("Failed to generate outline")
        try_count += 1

    messages.append({"role": "user", "content": prompt[1]})
    response = chat(
        model="gpt-4o-2024-08-06", 
        messages=messages,
        temperature=0.75,
        # max_tokens=15000
    )
    messages.append({"role": "assistant", "content": response})
    response_list.append(response)
    
    return prompt, messages, response_list

def generate_FT_outline(prompt):
    messages = [
        {"role": "user", "content": prompt[0]}
    ]
    try_count = 0
    response_list = []
    while try_count < 3:
        response = chat(
            model="ft:gpt-4o-2024-08-06:personal:better-outline:AeGKBUTq", 
            messages=messages
            # temperature=0.75,
            # max_completion_tokens=8000
        )
        if check_outline_completeness(response):
            messages.append({"role": "assistant", "content": response})
            response_list.append(response)
            break
        if try_count == 2:
            print(f"Failed to generate outline")
            print("response:")
            print(response)
            raise Exception("Failed to generate outline")
        try_count += 1

    messages.append({"role": "user", "content": prompt[1]})
    response = chat(
        model="ft:gpt-4o-2024-08-06:personal:better-outline:AeGKBUTq", 
        messages=messages,
        temperature=0.75,
        # max_tokens=15000
    )
    messages.append({"role": "assistant", "content": response})
    response_list.append(response)
    
    return prompt, messages, response_list