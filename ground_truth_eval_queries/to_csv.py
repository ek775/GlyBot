# transform eval queries into csv line by line
import pandas as pd

# dict of questions and ground truth answers
q_e_dict = {
    'question':[],
    'ground_truth':[]
}

# reading from text files
with open('./ground_truth_eval_queries/passage_start', 'r') as f:
    lines = f.readlines()
    for query in lines:
        prompt = "Complete the following paragraph: " + query.strip()
        q_e_dict['question'].append(prompt)

with open('./ground_truth_eval_queries/passage_continuation', 'r') as f:
    lines = f.readlines()
    for ground_truth in lines:
        prompt = ground_truth.strip()
        q_e_dict['ground_truth'].append(prompt)

# read other curated queries
with open('./ground_truth_eval_queries/curated_questions', 'r') as f:
    lines = f.readlines()
    for query in lines:
        prompt = query.strip()
        q_e_dict['question'].append(prompt)

with open('./ground_truth_eval_queries/curated_ground_truth', 'r') as f:
    lines = f.readlines()
    for ground_truth in lines:
        prompt = ground_truth.strip()
        q_e_dict['ground_truth'].append(prompt)

# write to csv

if len(q_e_dict['question'])==len(q_e_dict['ground_truth']):
    df = pd.DataFrame(q_e_dict)
    df.to_csv('./ground_truth_eval_queries/curated_queries.csv')