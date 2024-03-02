import json
import csv
from infer import *

def generate_prompt_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8-sig') as file:
        for line in file:
            try:
                data = json.loads(line)
                
                # Extract information from JSON
                question_stem = data['question']['stem']
                choices = data['question']['choices']
                answer_choices = '\n'.join([f"{choice['label']}. {choice['text']}" for choice in choices])

                # Create the prompt string
                prompt = f"**Question:**\n{question_stem}\n\n**Options:**\n{answer_choices}\n\nPlease provide the correct option (either A, B, C, or D) for this question."
                
                yield (prompt, question_stem, answer_choices)
            
            except json.JSONDecodeError:
                print("JSON decoding error")

def create_csv(json_file):
    result_file = f"result_{json_file.split('.')[0]}.csv"
    
    with open(result_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Formatted Question', 'Question', 'Choice', 'Inference Result'])
        
        for prompt, question_stem, answer_choices in generate_prompt_from_json(json_file):
            inference_result = do_inference(prompt)
            csv_writer.writerow([prompt, question_stem, answer_choices, inference_result])


json_files = ["ARC-Easy-Dev.jsonl", "ARC-Easy-Test.jsonl", "ARC-Easy-Train.jsonl", "ARC-Challenge-Dev.jsonl", "ARC-Challenge-Train.jsonl", "ARC-Challenge-Test.jsonl"]

for json_file in json_files:
    create_csv(json_file)
