import warnings
import random
import re
import logging
import os
import concurrent.futures
from typing import List

from rich import print
import time
from cohere import Client

from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

COHERE_API_KEY = os.environ["COHERE_API_KEY"]

logger = logging.getLogger(__name__)

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

def init_run(population: Population, model: Client, num_evals: int):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

    prompts = []

    for unit in population.units:    
        template= f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        prompts.append(template)
    
 
    results = model.batch_generate(prompts)

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(results):
        population.units[i].P = item[0].text

    _evaluate_fitness(population, model, num_evals)
    
    return population

def run_for_n(n: int, population: Population, model: Client, num_evals: int):
    """ Runs the genetic algorithm for n generations.
    """     
    p = population
    for i in range(n):  
        print(f"================== Population {i} ================== ")
        mutate(p, model)
        print("done mutation")
        _evaluate_fitness(p, model, num_evals)
        print("done evaluation")

    return p

def _evaluate_fitness(population: Population, model: Client, num_evals: int) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    #batch = random.sample(gsm8k_examples, num_evals)
    # instead of random, its better for reproducibility 
    batch = gsm8k_examples[:num_evals]

    elite_fitness = -1
    examples = []
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        # todo. model.batch this or multithread
        examples.append([unit.P + ' \n' + example['question'] for example in batch])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(examples)) as executor:
        future_to_fit = {executor.submit(model.batch_generate, example_batch,  temperature=0): example_batch for example_batch in examples}
        for future in concurrent.futures.as_completed(future_to_fit):
            example_batch = future_to_fit[future]  # Get the prompt corresponding to this future
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f"Exception: {exc}")


    # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
    # the LLM before further input Q.
    for unit_index, fitness_results in enumerate(results):
        for i, x in enumerate(fitness_results):
            valid = re.search(gsm.gsm_extract_answer(batch[i]['answer']), x[0].text)
            if valid:
                # 0.25 = 1 / 4 examples
                population.units[unit_index].fitness += (1 / num_evals)

            if unit.fitness > elite_fitness:
                # I am copying this bc I don't know how it might get manipulated by future mutations.
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness
    
    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population