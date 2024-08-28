# set-up ollama in env

import os
import pandas as pd
from openai import OpenAI
import gradio as gr


# Ollama client setup (same as before)
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

# Choose the model you want to use
model_name = "llama3:instruct"  # Or 'mistral'

# System prompt for RoA extraction (same as before)
system_prompt = """
You are given a label and description of a pharmaceutical product. 
Understand the given description and find out the route of administration (RoA) of the medicine. 
In pharmacology and toxicology, a route of administration is the way by which a drug, fluid, poison, or other substance is taken into the body. 
Routes of administration are generally classified by the location at which the substance is applied. Common examples include oral and intravenous administration. 
Strictly limit response to 1-2 words, which is the correct RoA inferred from the description. 
"""


def get_roa(label: str, drug_desc: str):
    """Extracts the route of administration (RoA) from a drug description."""
    drug_data = {"label": f"{label}", "description": f"{drug_desc}"}

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(drug_data)},
        ],
    )
    return resp.choices[0].message.content


def process_batch(drug_data: list):
    """Processes a batch of drug data (label, description) and extracts RoA."""
    return [get_roa(label, desc) for label, desc in drug_data]


def process_csv(input_file: str, label_column: str, desc_column: str, output_file: str):
    """
    Processes a CSV file containing drug labels and descriptions,
    extracts RoA, and saves the results to a new CSV file.
    """
    df = pd.read_csv(input_file)
    drug_data = list(zip(df[label_column], df[desc_column]))
    df["roa"] = process_batch(drug_data)
    df.to_csv(output_file, index=False)


def launch_gradio_app():
    """Launches a Gradio app for interactive RoA extraction."""

    iface = gr.Interface(
        fn=get_roa,
        inputs=[gr.Textbox("Drug Name"), gr.Textbox("Drug Description")],
        outputs="text",
        title="RoA (Route of Administration) Prediction",
        description="Enter the drug name and description to get the predicted Route of Administration (RoA).",
    )

    iface.launch()


def main(
    csv=False, input_file=None, label_column=None, desc_column=None, output_file=None
):
    """Main function to handle either CSV processing or Gradio app launch.

    Args:
        csv (bool): If True, processes a CSV file. If False, launches the Gradio app.
        input_file (str): Path to the input CSV file.
        label_column (str): Name of the column containing drug labels.
        desc_column (str): Name of the column containing drug descriptions.
        output_file (str): Path to the output CSV file.
    """
    if csv:
        if not all([input_file, label_column, desc_column, output_file]):
            raise ValueError(
                "When csv=True, you must provide input_file, label_column, desc_column, and output_file"
            )
        process_csv(input_file, label_column, desc_column, output_file)
    else:
        launch_gradio_app()


# Example usage:
# main(csv=True, input_file="path/to/input.csv", label_column="DrugLabel", desc_column="DrugDescription", output_file="path/to/output.csv")
# main()  # Launch Gradio app


if __name__ == "__main__":
    main()
