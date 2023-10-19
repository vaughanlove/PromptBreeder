import random
import re
import os
from pb.types import Population, EvolutionUnit
from langchain.llms.base import BaseLLM
from typing import List
from sentence_transformers import SentenceTransformer, util
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
from pb import gsm

from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.llms import VertexAI
from rich import print

load_dotenv()
aiplatform.init(project=os.getenv("PROJECT_ID")) # auth the google project

# wanted to abstract the model but it causes problems. For the time being just instantiating a model in here.
test_model = VertexAI(tempurature=1.0)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

# need below for estimation_distribution_mutation, not currently using.
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# print(model) 

# Direct Mutation mutators
def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, model: BaseLLM, **kwargs) -> EvolutionUnit:
    """Generates a new task-prompt P by concatenating the problem description D with the prompt 
    'a list of 100 hints:'. New task-prompt P is the first generated hint.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("zero_order_prompt_gen!")
    result = test_model(problem_description + " A list of 100 hints: ")
    # search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, result, re.DOTALL)
    if match: 
        # return the first match
        unit.P = match.group(1).strip()
    else: 
        unit.P = ""
    
    return unit 

def first_order_prompt_gen(unit: EvolutionUnit, model: BaseLLM, **kwargs) -> EvolutionUnit:
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P'
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("first_order_prompt_gen!")
    unit.P = test_model(unit.M + " " + unit.P) 
    return unit
    
# Estimation of Distribution Mutation - there is a variation of this called EDA rank
# and index mutation. I didn't implement it.
def estimation_distribution_mutation(unit: EvolutionUnit, population_units: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """ Provide a filtered and numbered list of the current population of task-prompts to the LLM and ask it to continue this list with new task-prompts.
    The List is filtered via ensuring that no two task-prompts have a score of >0.95 via BERT embedding cosine similarities.
    The List is randomly ordered.  

    NOTE: I am confused by this one. Does this mutate the entire population? What values of the continued list from the LLM do I use as prompts? randomly sampled?
    Not going to implement this one yet. Maybe should email someone. 
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    pass
def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], model: BaseLLM, **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("lineage_based_mutation")
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = ["{}. {}".format(i+1, x.P) for i, x in enumerate(elites)]
    unit.P = test_model(HEADING + ITEMS)
    
    return unit

# Hypermutation
def zero_order_hypermutation(unit: EvolutionUnit, problem_description: str, model: BaseLLM, **kwargs) -> EvolutionUnit:
    """ Concatenate the original problem_description to a randomly sampled thinking-style and feed it to the LLM to generate a new mutation-prompt.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("zero_order_hypermutation!")
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    unit.M = test_model(problem_description + RANDOM_THINKING_STYLE)
    return unit

def first_order_hypermutation(unit: EvolutionUnit, model: BaseLLM, **kwargs) -> EvolutionUnit:
    """ Concatenate the hyper-mutation prompt "Please summarize and improve the following instruction:"
    to a mutation-prompt to that the LLM generates a new mutation-prompt. This new mutation-prompt is then 
    instantly applied to the task-prompt of that unit.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("first_order_hypermutation!")

    HYPER_MUTATION_PROMPT="Please summarize and improve the following instruction: "
    unit.M = test_model(HYPER_MUTATION_PROMPT + unit.M)
    unit.P = test_model(unit.M + " " + unit.P)
    return unit 


# Lamarckian Mutation
def working_out_task_prompt(unit: EvolutionUnit, model: BaseLLM, **kwargs) -> EvolutionUnit:
    """ A 'lamarckian' mutation operator similar to instruction induction in APE.

    As far as I can understand, give it both the Q and A from the gsm8k dataset, 
    concatenated between 'I gave a friend an instruction and some advice. Here
    are the correct examples of his workings out ' and 'The instruction was: '
    The idea is to let the LLM reverse-engineer the task-prompt.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    print("working_out_task_prompt!")

    RANDOM_WORKING_OUT = random.sample(gsm8k_examples, 1)[0]
  
    unit.P = test_model("I gave a friend an instruction and some advice. Here are the correct examples of his workings out " + RANDOM_WORKING_OUT['question'] +  RANDOM_WORKING_OUT['answer'] + " The instruction was: ")
    return unit 

# Prompt crossover and context shuffling. These happen AFTER mutation operators. 
def prompt_crossover(**kwargs):
    """
    After a mutation operator is applied, 

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
def context_shuffling(**kwargs):
    """
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """

# omitting the estimation_distribution_mutation
MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    #estimation_distribution_mutation,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt
]

POST_MUTATORS = [
    prompt_crossover,
    context_shuffling
]

def mutate(population: Population, model: BaseLLM):
    """Select and apply a random mutator"""
    # steps
    # 1. parse through the population, grouping each evo unit by 2
    # 2. for each pair of evo units, using a uniform distribution, select a random mutator (of the 9)
    # 3. mutate and populate population.units

    # make a copy of population
    temporary_units = population.units.copy()

    for i in range(len(population.units) // 2):
        first_random_index = random.randrange(len(temporary_units))
        first_unit = population.units[first_random_index]
        temporary_units.pop(first_random_index)
    
        second_random_index = random.randrange(len(temporary_units))
        second_unit = population.units[second_random_index]
        temporary_units.pop(second_random_index)
        
        print("%"*50)
        print("First unit: \n")
        print(first_unit)
        print("%"*50)
        print("Second unit: \n")
        print(second_unit)
        # determine which unit has the higher fitness. Since I am currently testing and want to preserve the # of calls I am making to the LLM, there 
        # is a decent chance that I will hit equal fitness levels. in that case, first unit wins and second unit loses.
        
        # this is not great code
        FIRST_WON = False
        if population.units[first_random_index].fitness >=  population.units[second_random_index].fitness:
            # loser gets mutated.
            FIRST_WON = True
            input_unit = population.units[second_random_index].model_copy()
        else:
            input_unit = population.units[first_random_index].model_copy()
        
        data = {
            'unit' : input_unit,
            'model' : model,
            'elites' : population.elites,
            'problem_description': population.problem_description,
        }

        random_mutator = random.sample(MUTATORS, 1)[0]

        new_unit = random_mutator(**data)
        print("here")
        if FIRST_WON: 
            population.units[second_random_index] = new_unit
        else:
            population.units[first_random_index] = new_unit
