import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

os.environ["TORCH_USE_REENTRANT"] = "False"

# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
# ~ 3.3 Billion parameters
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "../data/raw/llm_training_data.jsonl"
OUTPUT_PATH = "../models/model"
SYSTEM_PROMPT = "you are him"

EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map = "auto",
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # model.to("cuda")

    return model, tokenizer

def get_lora_config():
    return LoraConfig(
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "down_proj", "up_proj"]
    )

def get_training_args():
    return TrainingArguments(
        output_dir = OUTPUT_PATH,
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate = LR,
        save_strategy = "epoch",
        logging_steps = 10,
        fp16 = True,
        bf16 = False,
        report_to = "tensorboard",
        ddp_find_unused_parameters = False,
        fsdp = None,
        optim = "paged_adamw_8bit"
    )

def setup_trainer():
    dataset = load_dataset("json", data_files = DATASET_PATH, split = "train")

    model, tokenizer = load_model_and_tokenizer()
    lora_config = get_lora_config()
    training_args = get_training_args()

    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        processing_class = tokenizer,
        peft_config = lora_config,
    )

    trainer.train()
    
    trainer.model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("Training completed")

def load_model_for_chat(base_model_id, adapter_path):
    print("Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.float16,
    )

    print(f"Loading base model ({base_model_id}) with 4-bit quantization...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config = bnb_config,
        device_map = "auto"
    )

    print(f"Attaching and merging adapter weights from {adapter_path}...")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() 
    model.eval()
    
    print("Ready")
    
    return model, tokenizer

def chat(model, tokenizer):
    os.system('cls')
    
    print("AI: Ask me anything")
    
    while True:
        try:
            user_input = input("\n>>> ")

            if user_input.lower() in ['quit', 'exit']:
                break
            
            formatted_input = (
                f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
                f"<|user|>\n{user_input}<|end|>\n"
                f"<|assistant|>"
            )
            
            inputs = tokenizer(
                formatted_input, 
                return_tensors = "pt", 
                padding = True
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = 512,
                    do_sample = True,
                    temperature = 0.7,
                    top_p = 0.95,
                    pad_token_id = tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(
                outputs[0], 
                skip_special_tokens = False
            )
            
            assistant_start_tag = "<|assistant|>"
            
            response = response.split(assistant_start_tag)[-1].split("<|end|>")[0].strip()
            
            print(f"\nAI: {response}")

        except Exception as e:
            print(f"\nerror: {e}")
            
            break

if __name__ == "__main__":
    setup_trainer()
    
    os.system('cls')
    
    try:
        model, tokenizer = load_model_for_chat(BASE_MODEL, OUTPUT_PATH)
        chat(model, tokenizer)
    except RuntimeError as e:
        print("EROR: ", e)