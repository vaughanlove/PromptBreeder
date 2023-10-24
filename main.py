from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse

from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.llms import VertexAI
from rich import print

load_dotenv() # load environment variables
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
parser.add_argument('-ts', '--num_thinking_styles', default=4)     
parser.add_argument('-e', '--num_evals', default=10)     
parser.add_argument('-n', '--simulations', default=10)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       

args = vars(parser.parse_args())

# parameters for our VertexAI model when mutating. temp=1 for randomness,
# allowing us to search more of the prompt space.
parameters = {
    "temperature": 1.0,  
    "max_output_tokens": 256, 
}

model = VertexAI(**parameters)

N = 4
tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set= thinking_styles[:int(args['num_thinking_styles'])]

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, model, int(args['num_evals']))

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, model=model, num_evals=int(args['num_evals']))

print("%"*80)
print("done processing! final gen:")
print(p)
