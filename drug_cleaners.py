import instructor
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import OpenAI
from tenacity import retry
from pydantic import BaseModel, Field

tqdm.pandas()

import os
import pandas as pd
import logging
from typing import List
from retry import retry
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ExtractDrugname(BaseModel):
    drugname_parts: List[str] = Field(
        description="Breakdown drug name and step-by-step list down identified parts"
    )
    reasoning: str = Field(
        description="Provide correct reasoning to explain which part should be the drug name required."
    )
    drug_name: str = Field(
        description="Correctly extracted drug name ONLY. No useless information should be provided."
    )

# System prompt for the LLM
system_prompt = """
You are given a raw drug name which may contain one or many parts of the drug name:
brand name, drug code, administration instructions, concentration, placebo information.
raw drug name might also be a combination of drugs.

Your task is to identify and extract the correct drug name ONLY. DO NOT provide any redundant information/explanation.

Important points to consider:
- if drug name is not given but drug code is given, provide complete drug code
- if no drug name is found, STRICTLY provide "NA"
- include all drugs if multiple drug names are found.

Find the raw drug name in single backticks (`).
"""

client = instructor.patch(
    OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama"),
    mode=instructor.Mode.JSON,
)

@retry
def process_drug_name(raw_drug_name: str, model_name: str):
    """Processes a single drug name using the specified model."""
    try:
        logger.info(f"Processing drug name: {raw_drug_name} with model: {model_name}")
        # Pull the Ollama model if it's not already loaded
        if not check_model_loaded(model_name):
            pull_ollama_model(model_name)

        resp = client.chat.completions.create(
            model=model_name,
            response_model=ExtractDrugname,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"raw_drug_name: `{raw_drug_name}`"},
            ],
        )

        if not isinstance(resp, ExtractDrugname):
            raise ValueError("Unexpected response format")
        return resp
    except Exception as e:
        logger.error(f"Error processing `{raw_drug_name}` with {model_name}: {e}")
        raise e

def process_batch(drug_names: list, model_name: str):
    """Processes a batch of drug names using the specified model."""
    logger.info(f"Starting batch processing with model: {model_name}")
    return [process_drug_name(name, model_name) for name in drug_names]

def process_csv(input_file: str, drug_column: str, output_file: str, models: list = ["mistral"]):
    """
    Processes a CSV file containing raw drug names and saves the cleaned
    drug names to a new CSV file, dynamically loading and unloading models.
    """
    logger.info(f"Reading input file: {input_file}")
    df = pd.read_excel(input_file)
    df = df.sample(50)

    for model in models:
        logger.info(f"Processing with model: {model}...")
        df[f"{model}_output"] = df[drug_column].progress_apply(
            lambda x: process_drug_name(x, model).drug_name
        )
        logger.info(f"Finished processing with model: {model}, unloading model.")
        # Free GPU memory after processing with each model
        unload_ollama_model(model)

    logger.info(f"Saving cleaned data to output file: {output_file}")
    df.to_csv(output_file, index=False)
    logger.info("Drug cleaning completed successfully!")

def check_model_loaded(model_name: str):
    """Checks if an Ollama model is loaded."""
    logger.debug(f"Checking if model {model_name} is loaded.")
    return os.system(f"ollama list | grep {model_name} > /dev/null") == 0

def pull_ollama_model(model_name: str):
    """Pulls an Ollama model."""
    logger.info(f"Pulling model: {model_name}")
    os.system(f"ollama pull {model_name}")

def unload_ollama_model(model_name: str):
    """Unloads an Ollama model."""
    logger.info(f"Unloading model: {model_name}")
    os.system(f"ollama run {model_name} --stop")

def main():
    """Main function to run the drug cleaning process."""
    input_file = "/content/Drug_cleaning_3k.xlsx"
    drug_column = "Drug"
    output_file = "/content/cleaned_drug_names_3k.csv"
    models = ['llama3.1:8b-instruct-q8_0']

    process_csv(input_file, drug_column, output_file, models)

if __name__ == "__main__":
    main()

main()

from google.colab import runtime
runtime.unassign()

