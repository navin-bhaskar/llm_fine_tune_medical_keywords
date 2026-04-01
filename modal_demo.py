import modal
app = modal.App("gradio-app")

web_image = modal.Image.debian_slim().pip_install("fastapi[standard]", "gradio", "torch", 
                                                "transformers", "bitsandbytes", "accelerate", "peft")

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "medical_keywords"
HF_USER_NAME = "navinbhaskar"


DATA_SET_USER_NAME = "navinbhaskar"
DATASET_NAME = f"{DATA_SET_USER_NAME}/medical_keywords_dataset"

RUN_NAME = "medical_keywords-2026-03-13_06.17.15"

PROJECT_RUN_NAME = f"medical_keywords-2026-03-13_06.17.15"
HUB_MODEL_NAME = f"{HF_USER_NAME}/{PROJECT_RUN_NAME}"
FINETUNED_MODEL = f"{HF_USER_NAME}/{PROJECT_RUN_NAME}"

secrets = [modal.Secret.from_name("huggingface-secret")]

@app.function(image=web_image, max_containers=1, secrets=secrets, gpu=GPU) # Keep max_containers=1 for statefulness
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    
    def predict(symptoms, age_group, duration_value, duration_unit, severity):
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
        
        # Get the HF token injected by Modal secrets
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or True 
        
        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto", token=hf_token
        )

        fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, token=hf_token)
        prompt = f"Instruction: Extract the relevant medical keywords from the following symptoms and patient details.\n\nInput:\nSymptoms: {symptoms}\nSeverity: {severity}\nDuration: {duration_value} {duration_unit}\nAge Group: {age_group}\n\nOutput:"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = fine_tuned_model.generate(inputs, max_new_tokens=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the content after "Output:"
        if "Output:" in result:
            result = result.split("Output:")[-1].strip()
        
        return result

    with gr.Blocks(title="Symptom to Medical Keywords") as demo:
        gr.Markdown("# Symptom to Medical Keywords Extractor")
        gr.Markdown("Enter patient symptoms and details to extract clinical keywords. *(Note: The keywords are extracted based on synthetic data used to fine-tune quantized Llama-3.2-3B model)*")
        
        with gr.Row():
            with gr.Column():
                symptoms = gr.Textbox(
                    lines=5, 
                    label="Symptoms", 
                    placeholder="e.g., patient complains of severe headache and nausea..."
                )
                with gr.Row():
                    age_group = gr.Radio(
                        choices=["child", "teen", "adult", "elderly"], 
                        label="Age Group", 
                        value="adult"
                    )
                    severity = gr.Radio(
                        choices=["mild", "moderate", "severe"], 
                        label="Severity", 
                        value="moderate"
                    )
                
                duration_value = gr.Slider(minimum=1, maximum=52, step=1, label="Duration", value=1)
                duration_unit = gr.Radio(
                    choices=["days", "weeks"], 
                    label="Duration Unit", 
                    value="days"
                )
                    
                submit_btn = gr.Button("Extract Keywords", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(label="Extracted Medical Keywords", lines=10)
                
        submit_btn.click(
            fn=predict,
            inputs=[symptoms, age_group, duration_value, duration_unit, severity],
            outputs=output_text
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")