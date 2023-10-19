from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.llms import VertexAI
from rich import print

load_dotenv() # load environment variables
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

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

N = 1
tp_set = mutation_prompts[:N]
mutator_set= thinking_styles[:2]
problem_description="Solve the math word problem, giving your answer as an arabic numeral."


p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=problem_description)
init_run(p, model)
run_for_n(n=5, population=p, model=model)

print("%"*80)
print("done processing! final gen:")
print(p)

exit()
