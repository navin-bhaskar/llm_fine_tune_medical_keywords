## Introduction
This project is a fine-tuned model for extracting medical keywords from symptoms and patient details.
This project mainly is for getting my hands dirty with LLM fine-tuning and deploying it on cloud using Modal.
The notebook used to analyze data can be found here: [train_medical_keyword.ipynb](train_medical_keyword.ipynb)

## Dataset
The dataset is a synthetic dataset of 100k medical keywords extracted from symptoms and patient details.

## Model
The model is a fine-tuned Llama-3.2-3B model for extracting medical keywords from symptoms and patient details.

## Deployment
The model is deployed on Modal for easy access and use.

## Finetuning on Google colab (with weights and biases monitoring)
https://colab.research.google.com/drive/1iWYxi9eMCCu6VC6LR1OcUToyiyKkyiF3?usp=sharing

## Usage
To test the model, update the HF_USER_NAME to your HF user name in `modal_demo.py` and then deploy the app using the modal CLI.
```bash
modal deploy modal_demo.py
```
Note: This will still use my finetuned HF model.
Demo: https://navin-bhaskar-5--gradio-app-ui.modal.run


