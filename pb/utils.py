import gsm
import re
import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.llms import VertexAI

load_dotenv() # load environment variables
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

model = VertexAI(tempurature=0.0)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

def evaluate_entire_dataset(prompt: str, max_examples=100):
    """"""
    points = 0

    batch = []
    answers = []
    for ex in gsm8k_examples[:max_examples]:
        batch.append(prompt + ' ' + ex['question'])
        answers.append(ex['answer'])

    results = model.batch(batch)

    assert len(results) == len(batch), "Something went wrong, these lengths should be the same."

    for i, result in enumerate(results):
        if re.search(gsm.gsm_extract_answer(answers[i]), result):
            points += 1

    print(points)
    print(points / max_examples)

evaluate_entire_dataset("""To solve a problem, follow these steps:\n\n1. Identify the problem.\n2. Gather
information about the problem.\n3. Brainstorm solutions to the problem.\n4. Evaluate the
solutions and choose the best one.\n5. Implement the solution.\n6. Evaluate the solution and make
adjustments as needed""")
        

    