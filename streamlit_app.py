import streamlit as st
import pandas as pd
from drug_cleaners import process_drug_name, process_csv as process_drug_names
from extract_epidemiology_ollama import (
    extract_epidemiology_info,
    process_csv as process_epidemiology,
)
from roa_extraction import get_roa, process_csv as process_roa

MODELS = ["llama2:7b-chat", "gemma:2b", "phi:latest"]


def process_csv(file, task, models, column_name):
    if task == "Drug Name Cleaning":
        output_file = "cleaned_drug_names.csv"
        process_drug_names(file.name, column_name, output_file, models)
    elif task == "Epidemiology Extraction":
        output_file = "extracted_epidemiology.csv"
        process_epidemiology(file.name, column_name, output_file, models)
    elif task == "RoA Extraction":
        output_file = "extracted_roa.csv"
        process_roa(
            file.name, column_name, column_name, output_file
        )  # Assuming same column for label and description

    return pd.read_csv(output_file)


def instant_trial(text, task, model):
    if task == "Drug Name Cleaning":
        result = process_drug_name(text, model)
        return f"Cleaned Drug Name: {result.cleaned_drug_name}\nReasoning: {result.reasoning}"
    elif task == "Epidemiology Extraction":
        result = extract_epidemiology_info(text, model)
        return f"Disease: {result.disease}\nPrevalence Rate: {result.prevalence_rate}\nIncidence Rate: {result.incidence_rate}\nMortality Rate: {result.mortality_rate}\nYear: {result.year}\nHospitalization Rate: {result.hospitalization_rate}\nKeywords: {result.keywords}\nChain of Thought: {result.chain_of_thought}"
    elif task == "RoA Extraction":
        result = get_roa(text, text)  # Assuming same text for label and description
        return f"Route of Administration: {result}"


st.title("Drug Information Extraction Tool")

mode = st.radio("Choose mode", ["CSV Processing", "Instant Trial"])

if mode == "CSV Processing":
    st.header("CSV Processing")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    task = st.radio(
        "Select Task",
        ["Drug Name Cleaning", "Epidemiology Extraction", "RoA Extraction"],
    )
    models = st.multiselect("Select Models", MODELS)
    column_name = st.text_input("Column Name to Process")

    if uploaded_file is not None and st.button("Process CSV"):
        with st.spinner("Processing..."):
            result_df = process_csv(uploaded_file, task, models, column_name)
        st.success("Processing complete!")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download processed CSV",
            data=csv,
            file_name="processed_output.csv",
            mime="text/csv",
        )

else:
    st.header("Instant Trial")
    text_input = st.text_area("Enter Text to Process")
    task = st.radio(
        "Select Task",
        ["Drug Name Cleaning", "Epidemiology Extraction", "RoA Extraction"],
    )
    model = st.selectbox("Select Model", MODELS)

    if st.button("Process Text"):
        with st.spinner("Processing..."):
            result = instant_trial(text_input, task, model)
        st.text_area("Result", result, height=300)

st.sidebar.info(
    "This app uses Ollama models to process drug-related information. Please ensure Ollama is running and the required models are available."
)
