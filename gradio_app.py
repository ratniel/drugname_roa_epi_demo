import gradio as gr
import pandas as pd
import drug_cleaners  # Assuming you have these files in the same directory
import roa_extraction
import extract_epidemiology_ollama
import os


def check_model_available(model_name):
    """Checks if an Ollama model is available locally."""
    return os.system(f"ollama list | grep {model_name} > /dev/null") == 0


def pull_model_if_needed(model_name):
    """Pulls an Ollama model if it's not available locally."""
    if not check_model_available(model_name):
        print(f"Pulling model: {model_name}...")
        os.system(f"ollama pull {model_name}")


def process_data(task, input_type, model_names, input_file, text_input, label_input, desc_input):
    """Processes data based on the selected task, input type, and model."""

    if input_type == "from csv":
        selected_models = [model for model, checked in model_names if checked]
        for model in selected_models:
            pull_model_if_needed(model)
    else:  # Instant trial
        selected_model = model_names  # Get the single selected model
        pull_model_if_needed(selected_model)

    if task == "drug name cleaning":
        if input_type == "from csv":
            drug_cleaners.process_csv(
                input_file.name, "Drug", "cleaned_drug_names.csv", selected_models
            )
            return "cleaned_drug_names.csv"
        else:
            return drug_cleaners.process_drug_name(
                text_input, selected_model
            ).drug_name

    elif task == "roa extraction":
        if input_type == "from csv":
            roa_extraction.process_csv(
                input_file.name, "Drug Label", "Drug Description", "roa_extracted.csv"
            )
            return "roa_extracted.csv"
        else:
            return roa_extraction.get_roa(label_input, desc_input)

    elif task == "epidemiology extraction":
        if input_type == "from csv":
            extract_epidemiology_ollama.process_csv(
                input_file.name,
                "Result Text",
                "epidemiology_extracted.csv",
                selected_models,
            )
            return "epidemiology_extracted.csv"
        else:
            return extract_epidemiology_ollama.extract_epidemiology_info(
                text_input, selected_model
            )

    else:
        return "Invalid task selected."


def launch_gradio_interface():
    """Launches the Gradio interface."""
    available_models = [
        ("llama3.1:8b-instruct-q8_0", True),
        ("gemma2:9b-instruct-q8_0", True),
        ("phi3.5:3.8b-mini-instruct-q8_0", True),
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            task = gr.Dropdown(
                choices=[
                    "drug name cleaning",
                    "roa extraction",
                    "epidemiology extraction",
                ],
                label="Select Task",
            )
            input_type = gr.Radio(
                choices=["from csv", "instant trial"], label="Input Type"
            )

        with gr.Row():
            model_section = gr.Column(visible=False)
            with model_section:
                model_names_csv = gr.CheckboxGroup(
                    choices=available_models,
                    label="Select Models (check to enable)",
                    visible=False
                )
                model_name_instant = gr.Radio(
                    choices=[model[0] for model in available_models],
                    label="Select Model",
                    visible=False
                )

        with gr.Row():
            input_section = gr.Column(visible=False)
            with input_section:
                input_file = gr.File(label="Upload CSV File", visible=False)
                text_input = gr.Textbox(
                    label="Enter Text", visible=False, lines=5
                )
                label_input = gr.Textbox(label="Enter Drug Label", visible=False)
                desc_input = gr.Textbox(
                    label="Enter Drug Description", visible=False, lines=3
                )

        with gr.Row():
            output_text = gr.Textbox(label="Output")

        def update_visibility(task_choice, input_type_choice):
            csv_visible = input_type_choice == "from csv"
            instant_visible = input_type_choice == "instant trial"
            
            return {
                model_section: gr.update(visible=True),
                model_names_csv: gr.update(visible=csv_visible),
                model_name_instant: gr.update(visible=instant_visible),
                input_section: gr.update(visible=True),
                input_file: gr.update(visible=csv_visible),
                text_input: gr.update(visible=instant_visible and task_choice == "epidemiology extraction"),
                label_input: gr.update(visible=instant_visible and task_choice == "roa extraction"),
                desc_input: gr.update(visible=instant_visible and task_choice == "roa extraction"),
            }

        input_type.change(update_visibility, inputs=[task, input_type], 
                         outputs=[model_section, model_names_csv, model_name_instant, input_section, input_file, text_input, label_input, desc_input])

        demo.load(
            process_data,
            inputs=[
                task,
                input_type,
                model_names_csv,  # Pass checkbox group for CSV input
                input_file,
                text_input,
                label_input,
                desc_input,
            ],
            outputs=output_text,
        )

        demo.load(
            process_data,
            inputs=[
                task,
                input_type,
                model_name_instant,  # Pass radio button for instant trial
                input_file,
                text_input,
                label_input,
                desc_input,
            ],
            outputs=output_text,
        )

    demo.launch()


if __name__ == "__main__":
    launch_gradio_interface()