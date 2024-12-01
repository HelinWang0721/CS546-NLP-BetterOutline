import random
import numpy as np
from typing import TypedDict, List

def sample_general_one(g) -> str:
    ret = g["fixed_words"]
    if len(g["optional_words"]) <= 0:
        return ret
    
    for o in g["optional_words"]:
        sample_size = 1
        if "max_num" in o and "min_num" in o:
            sample_size = random.randint(o["min_num"], o["max_num"])
            np.clip(sample_size, 0, len(o["words"]))
        tmp = list(np.random.choice(o["words"], sample_size, replace=False))
        ret = ret + " " + " ".join(tmp)
    return ret

general_weights = None
def sample_general(general_list) -> str:
    global general_weights
    if general_weights is None:
        weights = [g["weight"] for g in general_list]
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        general_weights = weights
    return sample_general_one(np.random.choice(general_list, p=general_weights))


def sample_optional(num, optional_words) -> str:
    # List[OptionalWords]
    weights = [1 if "weight" not in o else o["weight"] for o in optional_words]
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    chosen_optional = []
    while len(chosen_optional) < num:
        chosen_optional_list = np.random.choice(optional_words, p=weights)
        chosen_word = chosen_optional_list["words"][np.random.randint(len(chosen_optional_list["words"]))]
        if chosen_word not in chosen_optional:
            chosen_optional.append(chosen_word)
    return " ".join(chosen_optional)
