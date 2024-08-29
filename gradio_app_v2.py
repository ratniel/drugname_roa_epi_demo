import gradio as gr
import pandas as pd
from drug_cleaners import process_csv as process_drug_names
from extract_epidemiology_ollama import process_csv as process_epidemiology
from roa_extraction import process_csv as process_roa
import os

MODELS = ["llama3.1:8b-instruct-q8_0", "gemma2:9b-instruct-q8_0", "phi:latest"]


def process_csv(file, task, model, column_name):
    if task == "Drug Name Cleaning":
        output_file = "cleaned_drug_names.csv"
        process_drug_names(file.name, column_name, output_file, [model])
    elif task == "Epidemiology Extraction":
        output_file = "extracted_epidemiology.csv"
        process_epidemiology(file.name, column_name, output_file, [model])
    elif task == "RoA Extraction":
        output_file = "extracted_roa.csv"
        process_roa(
            file.name, column_name, column_name, output_file
        )  # Assuming same column for label and description

    return output_file


def instant_trial(text, task, model):
    if task == "Drug Name Cleaning":
        from drug_cleaners import process_drug_name

        result = process_drug_name(text, model)
        return f"Cleaned Drug Name: {result.cleaned_drug_name}\nReasoning: {result.reasoning}"
    elif task == "Epidemiology Extraction":
        from extract_epidemiology_ollama import extract_epidemiology_info

        result = extract_epidemiology_info(text, model)
        return f"Disease: {result.disease}\nPrevalence Rate: {result.prevalence_rate}\nIncidence Rate: {result.incidence_rate}\nMortality Rate: {result.mortality_rate}\nYear: {result.year}\nHospitalization Rate: {result.hospitalization_rate}\nKeywords: {result.keywords}\nChain of Thought: {result.chain_of_thought}"
    elif task == "RoA Extraction":
        from roa_extraction import get_roa

        result = get_roa(text, text)  # Assuming same text for label and description
        return f"Route of Administration: {result}"


def launch_app():
    with gr.Blocks() as app:
        gr.Markdown("# Drug Information Extraction Tool")

        with gr.Tab("CSV Processing"):
            csv_file = gr.File(label="Upload CSV File")
            csv_task = gr.Radio(
                ["Drug Name Cleaning", "Epidemiology Extraction", "RoA Extraction"],
                label="Select Task",
            )
            csv_models = gr.CheckboxGroup(choices=MODELS, label="Select Models")
            csv_column = gr.Textbox(label="Column Name to Process")
            csv_submit = gr.Button("Process CSV")
            csv_output = gr.File(label="Download Processed CSV")

            csv_submit.click(
                process_csv,
                inputs=[csv_file, csv_task, csv_models, csv_column],
                outputs=csv_output,
            )

        with gr.Tab("Instant Trial"):
            instant_text = gr.Textbox(label="Enter Text to Process")
            instant_task = gr.Radio(
                ["Drug Name Cleaning", "Epidemiology Extraction", "RoA Extraction"],
                label="Select Task",
            )
            instant_model = gr.Radio(choices=MODELS, label="Select Model")
            instant_submit = gr.Button("Process Text")
            instant_output = gr.Textbox(label="Result")

            instant_submit.click(
                instant_trial,
                inputs=[instant_text, instant_task, instant_model],
                outputs=instant_output,
            )

    # For Google Colab
    try:
        from google.colab import output

        output.serve_kernel_port_as_window(8080)
        app.queue().launch(server_port=7860, share=True, show_error=True, debug=True)
    except ImportError:
        # Fallback for non-Colab environments
        app.queue().launch(share=True, debug=True)


if __name__ == "__main__":
    launch_app()
