import os
import warnings
import random
import json
import re
from typing import List

from dotenv import load_dotenv
import pydantic
from google.cloud import aiplatform
from langchain.llms import vertexai
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from . import mutation_prompts
from . import thinking_styles
from . import gsm

# setup 
# ignore the langchain warning
warnings.filterwarnings("ignore", message="Importing llm_cache from langchain root ")
load_dotenv() # load environment variables
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

# parameters for our VertexAI model.
parameters = {
    "temperature": 0,  # Temperature controls the degree of randomness in token selection.
    "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
    "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
}

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

model = vertexai.VertexAI(**parameters)


class EvolutionUnit(pydantic.BaseModel):
    """ A individual unit of the overall population.

    Attributes:
        'T': the thinking_style.

        'M': the mutation_prompt.

        'P': the task_prompt.
        
        'fitness': the estimated performance of the unit.
        
        'history': historical prompts for analysis. 
    """
    T: str
    M: str
    P: str
    fitness: float
    history: List[str]

class Population(pydantic.BaseModel):
    """ Population model that holds the age of the population, its size, and a list of individuals.
    
    Attributes:
        'size' (int): the size of the population.

        'age' (int): the age of the population.
        
        'units' (List[EvolutionUnit]): the individuals of a population.
    """
    size: int
    age: int
    problem_description: str
    units: List[EvolutionUnit]


def create_population(size: int, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': size,
        'age': 0,
        'problem_description' : problem_description,
        'units': [EvolutionUnit(**{
            'T' : random.choice(thinking_styles.thinking_styles), 
            'M' : random.choice(mutation_prompts.mutation_prompts),
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for _ in range(size)]
    }

    return Population(**data)

def init_run(population: Population):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    
    
    batch = []
    for unit in population.units:
        batch.append({'T': unit.T, 'M' : unit.M, 'D' : population.problem_description})

    template= "{T} {M} INSTRUCTION: {D} INSTRUCTION MUTANT = "
    prompt=PromptTemplate(
        template=template,
        input_variables=["T", "M", "D"],
    )

    initialization_chain = prompt | model | StrOutputParser()
    response = initialization_chain.batch(batch)

    assert len(response) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(response):
        population.units[i].P = item

    return population

def run_for_n(n: int):
    """ Runs the genetic algorithm for n generations.
    """
    pass

def _evaluate_fitness(population: Population) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer.
    batch = random.sample(gsm8k_examples, 10)
    
    for unit in population.units:
        unit.fitness = 0
        # todo. batch this
        for example in batch:
            result = model(example['question'] + ' ' + unit.P)
            print(result)
            valid = re.search(gsm.gsm_extract_answer(example['answer']), result)
            if valid:
                print("correct answer.")
                unit.fitness += 0.1
