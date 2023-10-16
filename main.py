import os
import warnings

from google.cloud import aiplatform
from langchain.llms import VertexAI

from dotenv import load_dotenv

# ignore the langchain warning
warnings.filterwarnings("ignore", message="Importing llm_cache from langchain root ")

# load environment variables
load_dotenv()

# authenticate the google project
aiplatform.init(project=os.getenv("PROJECT_ID"))

parameters = {
    "temperature": 0.4,  # Temperature controls the degree of randomness in token selection.
    "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
    "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
}

model = VertexAI(**parameters)

thinking_style = "Let's think step by step"
mutation_prompt = "Change this instruction to make it more fun"
problem_description = "Solve this math word problem"

print(f"D: {problem_description}")
print(f"T: {thinking_style}")
print(f"M: {mutation_prompt}")


inputtext= f"""{thinking_style} {mutation_prompt} INSTRUCTION: {problem_description} INSTRUCTION MUTANT = """

print("P = D + T + M")

response = model.predict(
    inputtext,
)

print(f"P' = LLM(P): {response}")

# This is when one of the 9 mutators mutates the mutator. 