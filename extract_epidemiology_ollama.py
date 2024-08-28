import os
import pandas as pd
from openai import OpenAI
import instructor
from typing import List
from pydantic import BaseModel, Field
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


# Ollama client setup
client = instructor.patch(
    OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama"),
    mode=instructor.Mode.JSON,
)

system_prompt = """
You are a highly skilled medical researcher specializing in epidemiology. 
You are tasked with analyzing medical texts to extract crucial statistical information about diseases.

You will be provided with excerpts from medical articles. Your job is to carefully examine these texts and identify specific epidemiological data points, including prevalence rates, incidence rates, mortality rates, and hospitalization rates, along with the year the data refers to and the specific disease being discussed.

Pay close attention to numerical values and their context within the text. Focus on identifying rates expressed as percentages, proportions, or per population figures (e.g., per 100,000 people). 

If a specific data point is not found in the text, indicate it as "NA".

Your output should be well-structured and easy to understand.
"""


class EpidemiologyData(BaseModel):
    chain_of_thought: str = Field(
        ...,
        description="Detailed reasoning steps used to arrive at the extracted information. Explain how you identified each piece of information and why you believe it is accurate.",
    )
    keywords: List[str] = Field(
        ...,
        description="List of relevant keywords found in the text that indicate epidemiological data or the disease being discussed.",
    )
    prevalence_rate: str = Field(
        default="NA",
        description="Rate of people with the disease at a specific time. Express as a percentage or proportion. If not found, indicate 'NA'.",
    )
    incidence_rate: str = Field(
        default="NA",
        description="New cases of the disease in a specific time period (e.g., per year). Express as a rate per population (e.g., per 100,000 people). If not found, indicate 'NA'.",
    )
    mortality_rate: str = Field(
        default="NA",
        description="Rate of mortality due to the disease. Express as a percentage or proportion. If not found, indicate 'NA'.",
    )
    year: int = Field(
        default=0,
        description="Year the data refers to. If the year is not explicitly mentioned, try to infer it from the context. If not found or cannot be inferred, indicate 0.",
    )
    hospitalization_rate: str = Field(
        default="NA",
        description="Rate of people hospitalized due to the disease. Express as a percentage or proportion. If not found, indicate 'NA'.",
    )
    disease: str = Field(
        description="The specific disease being discussed in the text. If not explicitly mentioned or unclear, indicate 'NA'.",
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_epidemiology_info(text: str, model_name: str) -> EpidemiologyData:
    """Extracts epidemiology information from a text using the specified model."""
    try:
        # Pull the Ollama model if it's not already loaded
        if not check_model_loaded(model_name):
            pull_ollama_model(model_name)

        response: EpidemiologyData = client.chat.completions.create(
            model=model_name,  # Use the model name directly
            response_model=EpidemiologyData,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        return response
    except Exception as e:
        print(f"Error processing text with {model_name}: {e}")
        raise e


def process_csv(
    input_file: str, text_column: str, output_file: str, models: List[str]
):
    """
    Processes a CSV file, extracts epidemiology information using multiple
    models, and saves the results to a new CSV.
    """
    df = pd.read_csv(input_file)

    for model in models:
        print(f"Processing with {model}...")
        # Extract information and add it to the DataFrame
        df[f"{model}_prevalence_rate"] = ""
        df[f"{model}_incidence_rate"] = ""
        df[f"{model}_hospitalization_rate"] = ""
        df[f"{model}_mortality_rate"] = ""

        for index, row in df.iterrows():
            try:
                response = extract_epidemiology_info(row[text_column], model)
                df.loc[index, f"{model}_prevalence_rate"] = response.prevalence_rate
                df.loc[index, f"{model}_incidence_rate"] = response.incidence_rate
                df.loc[index, f"{model}_hospitalization_rate"] = response.hospitalization_rate
                df.loc[index, f"{model}_mortality_rate"] = response.mortality_rate
            except Exception as e:
                print(f"Error processing row {index} with {model}: {e}")
                df.loc[index, f"{model}_prevalence_rate"] = "Error"
                df.loc[index, f"{model}_incidence_rate"] = "Error"
                df.loc[index, f"{model}_hospitalization_rate"] = "Error"
                df.loc[index, f"{model}_mortality_rate"] = "Error"
                raise e

        # Add a blank column after each model's output
        df[f"{model}_blank"] = ""

    df.to_csv(output_file, index=False)

def check_model_loaded(model_name: str):
    """Checks if an Ollama model is loaded."""
    return os.system(f"ollama list | grep {model_name} > /dev/null") == 0


def pull_ollama_model(model_name: str):
    """Pulls an Ollama model."""
    os.system(f"ollama pull {model_name}")

def unload_ollama_model(model_name: str):
    """Unloads an Ollama model to free GPU memory."""
    os.system(f"ollama run {model_name} --stop")

def main():
    """Main function to run the epidemiology information extraction."""
    input_file = input("Enter the path to the input CSV file: ")
    text_column = input(
        "Enter the name of the column containing the medical text: "
    )
    output_file = input("Enter the path to the output CSV file: ")
    selected_models = input(
        "Enter the models to use (separated by commas, default: mistral): "
    )
    models = selected_models.split(",") if selected_models else ["mistral"]

    process_csv(input_file, text_column, output_file, models)

    # Unload models after processing
    for model in models:
        unload_ollama_model(model)

    print("Epidemiology information extraction completed successfully!")


if __name__ == "__main__":
    main()