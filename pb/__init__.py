import warnings
import random
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from rich import print
import time

from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

# setup 
# ignore the langchain warning
warnings.filterwarnings("ignore", message="Importing llm_cache from langchain root ")
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

def init_run(population: Population, model: BaseLLM):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

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

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

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

def _evaluate_fitness(population: Population, model: BaseLLM, batch_size=10) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    # create the testing batches - batch_size per unit
    # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
    # the LLM before further input Q.    
    answers = []

    # for each unit, take batch_size examples and append that unit's prompt. save the answers for later.
    logger.info(f"start threading")

    threads = []
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for unit in population.units:
            logger.info(f"opening thread for: {unit.P}")

            batch = []
            for example in random.sample(gsm8k_examples, batch_size):
                batch.append(example['question'] + " " + unit.P)
                answers.append(example['answer'])

            threads.append(executor.submit(model.batch, batch))

        for task in as_completed(threads):
            results.extend(task.result())
    
    logger.info(f"done threading: length of results: {len(results)}")

    # threading to heavily accelerate. hopefully this returns in the correct order.
    
    #results = model.batch(batches)     # feed it through the LLM
    elite_fitness = -1
    # for each unit, evaluate the results compared to the previously saved answers.
    for i, unit in enumerate(population.units):
        unit.fitness = 0
        eval_batch = results[i*batch_size:i*batch_size + batch_size]
        for j, x in enumerate(eval_batch):
            valid = re.search(gsm.gsm_extract_answer(answers[i*batch_size + j]), x)
            if valid:
                # 0.25 = 1 / 4 examples
                unit.fitness += (1 / batch_size)
            if unit.fitness > elite_fitness:
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness
    
    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population