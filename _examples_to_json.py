import json
from sentence_transformers import InputExample

def input_example_to_dict(input_example):
    return {
        'texts': input_example.texts,
        'label': input_example.label if hasattr(input_example, 'label') else None
    }

def save_examples_to_json(train_examples, file_path):
    examples_dict = [input_example_to_dict(example) for example in train_examples]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(examples_dict, f, ensure_ascii=False, indent=4)

def load_examples_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        examples_dict = json.load(f)
    return [InputExample(texts=example['texts'], label=example['label']) for example in examples_dict]


# Save to JSON
#save_examples_to_json(train_examples, 'train_examples.json')

# Load from JSON
#loaded_train_examples = load_examples_from_json('train_examples.json')
