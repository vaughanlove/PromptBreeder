from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging

from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.llms import VertexAI
from rich import print

load_dotenv() # load environment variables
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# parameters for our VertexAI model when evaluating questions.
cold_parameters = {
    "temperature": 0.0,  
    "max_output_tokens": 256,  
}

# parameters for our VertexAI model when mutating. temp=1 for randomness,
#  allowing us to search more of the prompt space.
parameters = {
    "temperature": 1.0,  
    "max_output_tokens": 256, 
}

cold_model = VertexAI(**cold_parameters)
model = VertexAI(**parameters)

N = 2
tp_set = mutation_prompts[:N]
mutator_set= thinking_styles[:N]
problem_description="Solve the math word problem, giving your answer as an arabic numeral."

logger.info(f'You are prompt-optimizing for the problem: {problem_description}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=problem_description)
logger.info(f'Generating the initial prompts...')
init_run(p, model)
logger.info(f'Starting the genetic algorithm...')
run_for_n(n=10, population=p, model=cold_model)

print("%"*80)
print("done processing! final gen:")
print(p)
