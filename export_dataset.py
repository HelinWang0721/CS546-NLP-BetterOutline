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

def outline_generate(input_file, output_file):

    try:
        current_data = load_input_json(output_file)
        start_index = len(current_data)
    except FileNotFoundError:
        current_data = []
        start_index = 0
        
    input_data = load_input_json(input_file)

    while start_index < len(input_data):
        try:
            # load the existing list that save every output entry in iteration
            current_data = load_input_json(output_file)    
        except FileNotFoundError:
            # if the file does not exist, create a new list
            current_data = []
        
        m_words = input_data[start_index]["m_words"]
        o_words = input_data[start_index]["o_words"]

        prompt, _, response = og.generate_outline(m_words, o_words)

        output_entry = {
            "m_words": m_words,
            "o_words": o_words,
            "prompt": [prompt[0], prompt[1]],
            "response": [response[0], response[1]],
        }
        current_data.append(output_entry)
        save(current_data, output_file)
        start_index += 1


outline_generate("tasks.json", "output_file.json")














