import warnings
import random
import re
from typing import List

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from rich import print


from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

# setup 
# ignore the langchain warning
warnings.filterwarnings("ignore", message="Importing llm_cache from langchain root ")

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

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
        'elites' : [],
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
    
    _evaluate_fitness(population, model)
    return population

def run_for_n(n: int, population: Population, model: BaseLLM):
    """ Runs the genetic algorithm for n generations.
    """     
    p = population
    for i in range(n):  
        print(f"================== Population {i} ================== ")
        mutate(p, model)
        print("done mutation")
        _evaluate_fitness(p, model)
        print("done evaluation")

    return p

def _evaluate_fitness(population: Population, model: BaseLLM, batch_size=4) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    batch = random.sample(gsm8k_examples, batch_size)
    
    elite_fitness = -1
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        # todo. model.batch this or multithread
        examples = [unit.P + ' ' + example['question'] for example in batch]
        # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
        # the LLM before further input Q.            
        results = model.batch(examples)
        for i, x in enumerate(results):
            valid = re.search(gsm.gsm_extract_answer(batch[i]['answer']), x)
            if valid:
                # 0.25 = 1 / 4 examples
                unit.fitness += (1 / batch_size)

            if unit.fitness > elite_fitness:
                # I am copying this bc I don't know how it might get manipulated by future mutations.
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness
    
    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    return population