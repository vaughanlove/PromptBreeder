import os
import warnings
import random
import json
import re
from typing import List

import pydantic

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from . import gsm

# setup 
# ignore the langchain warning
warnings.filterwarnings("ignore", message="Importing llm_cache from langchain root ")

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')
 
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

def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description' : problem_description,
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)

def init_run(population: Population, model: BaseLLM):
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

def run_for_n(n: int, population: Population, model: BaseLLM):
    """ Runs the genetic algorithm for n generations.
    """
    for i in range(n):
        print("Population {i}")
        p = _evaluate_fitness(population, model)
        # in here is when I pick 1 of the 9 potential mutation operators.
        for i, x in enumerate(population.units):
            population.units[i] = random_mutator(x)

    return p

def _evaluate_fitness(population: Population, model: BaseLLM, batch_size=4) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    batch = random.sample(gsm8k_examples, batch_size)
    
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        # todo. model.batch this or multithread
        for example in batch:
            # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
            # the LLM before further input Q.
            result = model(unit.P + ' ' + example['question'])
            valid = re.search(gsm.gsm_extract_answer(example['answer']), result)
            if valid:
                # 0.25 = 1 / 4 examples
                unit.fitness += (1 / batch_size)
    return population