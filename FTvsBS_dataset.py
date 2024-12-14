import json
from tracemalloc import start
import outline_gen as og

def load_input_json(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save(data_list, filename):

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def outline_generate(input_file, BS_file, FT_file):

    try:
        current_BS_data = load_input_json(BS_file)
        current_FT_data = load_input_json(FT_file)
        start_index = len(current_FT_data)
    except FileNotFoundError:
        current_BS_data = []
        current_FT_data = []
        start_index = 0
        
    input_data = load_input_json(input_file)

    # while start_index < len(input_data):
    while start_index < 10:
        try:
            # load the existing list that save every output entry in iteration
            current_BS_data = load_input_json(BS_file) 
            current_FT_data = load_input_json(FT_file)   
        except FileNotFoundError:
            # if the file does not exist, create a new list
            current_BS_data = []
            current_FT_data = []
        
        m_words = input_data[start_index]["m_words"]
        o_words = input_data[start_index]["o_words"]

        promptBS, _, responseBS = og.generate_BS_outline(m_words, o_words)
        promptFT, _, responseFT = og.generate_FT_outline(promptBS)

        BS_output_entry = {
            "m_words": m_words,
            "o_words": o_words,
            "prompt": [promptBS[0], promptBS[1]],
            "response": [responseBS[0], responseBS[1]],
        }
        FT_output_entry = {
            "m_words": m_words,
            "o_words": o_words,
            "prompt": [promptFT[0], promptFT[1]],
            "response": [responseFT[0], responseFT[1]],
        }

        current_BS_data.append(BS_output_entry)
        current_FT_data.append(FT_output_entry)
        save(current_BS_data, BS_file)
        save(current_FT_data, FT_file)
        start_index += 1


outline_generate("eval_tasks.json", "BS_file.json", "FT_file.json")