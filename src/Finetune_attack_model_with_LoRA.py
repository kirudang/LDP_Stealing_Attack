import os
# Set the HF_HOME environment variable to point to the desired cache location
os.environ["HF_TOKEN"] = "Your_Huggingface_Token"
# Specify the directory path 
cache_dir = 'Your_Cache_Directory'
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = cache_dir

import matplotlib.pyplot as plt
import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, HfArgumentParser, TrainingArguments, BitsAndBytesConfig
import json
import argparse
import torch.nn as nn
from typing import Optional
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from trl import SFTTrainer
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def parse_args():
    ### Keep the same parameters in the Add_noise_embedding.py to load the correct noisy embeddings
    parser = argparse.ArgumentParser(description="Finetune attack model with LoRA")
    parser.add_argument('--mode', type=str, default='watermark', help="Mode of finetune: 'no_watermark' or 'watermark'")
    parser.add_argument('--mechanism', type=str, default='lap', help='Mechanism of noise: Laplace')
    parser.add_argument('--eps', type=float, default=10, help='Epsilon')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean')
    ### Change below parameters for your fine-tuning experiment
    parser.add_argument('--type_of_text', type=str, default='input_output', help='Text column to use for finetuning in your dataset')
    parser.add_argument('--num_training_data', type=int, default=10000, help='Number of data to use for training, default is 10000')
    parser.add_argument('--model_name', type=str, default='llama2', help='Model to use for finetuning')
    parser.add_argument('--wm_mode', type=str, default='icml', help="Watermark type to attack on. The others are: 'rdf', 'SIR', 'semstamp'")
    parser.add_argument('--std', type=float, default=0.01, help='Noise scale. It is the --scale in Add_noise_embedding.py')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    return parser.parse_args()


@dataclass
class ScriptArguments:
    """
    Define the arguments used in this script.
    """
    use_8_bit: Optional[bool] = field(default=False, metadata={"help": "use 8 bit precision"})
    use_4_bit: Optional[bool] = field(default=False, metadata={"help": "use 4 bit precision"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    use_multi_gpu: Optional[bool] = field(default=True, metadata={"help": "use multi GPU"})
    use_adapters: Optional[bool] = field(default=True, metadata={"help": "use adapters"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "input batch size"})
    max_seq_length: Optional[int] = field(default=400, metadata={"help": "max sequence length"})
    optimizer_name: Optional[str] = field(default="adamw_hf", metadata={"help": "Optimizer name"})


def load_data(file_path, type_of_text):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return [item[type_of_text] for item in data]

def create_dataset(data):
    df = pd.DataFrame({"text": data})
    return Dataset.from_pandas(df)

def get_file_paths(mode, wm_mode, model_name):
    base_path = 'Text_generation/'
    paths = {
        'no_watermark': {
            'llama2': ('Normal_output/output_llama2_train_final.json', 'Normal_output/output_llama2_test_final.json'),
            'mistral7b': ('Normal_output/output_mistral_train_final.json', 'Normal_output/output_mistral_test_final.json')
        },
        'watermark': {
            'icml': {
                'llama2': ('ICML_WM/output_llama2_watermarked_train_final.json', 'ICML_WM/output_llama2_watermarked_test_final.json'),
                'mistral7b': ('ICML_WM/output_mistral_ICML_train_final.json', 'ICML_WM/output_mistral_ICML_test_final.json')
            },
            'rdf': {
                'llama2': ('Distortion_free_WM/output_rdf_llama2_train_final.json', 'Distortion_free_WM/output_rdf_llama2_test_final.json'),
                'mistral7b': ('Distortion_free_WM/output_rdf_mistral7b_train_final.json', 'Distortion_free_WM/output_rdf_mistral7b_test_final.json')
            },
            'SIR': {
                'llama2': ('SIR/output_llama2_SIR_train_10000.json', 'SIR/output_llama2_SIR_test_2000.json'),
                'mistral7b': ('SIR/output_mistral7b_SIR_train_10000.json', 'SIR/output_mistral7b_SIR_test_2000.json')
            },
            'semstamp': {
                'llama2': ('Semstamp_WM/output_Semstamp_llama2_train_10000.json', 'Semstamp_WM/output_Semstamp_llama2_test_2000.json'),
                'mistral7b': ('Semstamp_WM/output_Semstamp_mistral7b_train_10000.json', 'Semstamp_WM/output_Semstamp_mistral7b_test_2000.json')
            }
        }
    }
    if mode == 'no_watermark':
        return base_path + paths[mode][model_name][0], base_path + paths[mode][model_name][1]
    else:
        return base_path + paths[mode][wm_mode][model_name][0], base_path + paths[mode][wm_mode][model_name][1]

def get_new_model_path(mode, wm_mode, mechanism, mean, std, model_name, num_epochs, learning_rate_, type_of_text, num_training_data):
    mean_str = str(mean)
    std_str = str(std)
    num_epochs_str = str(num_epochs)
    learning_rate_str = str(learning_rate_)
    num_training_data_str = str(num_training_data)

    if mode == 'no_watermark':
        return f"./Finetune_newmodels/save_noS2M_{mechanism}_{mean_str}_{std_str}_{model_name}/no_watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"
    else:
        return f"./Finetune_newmodels/save_noS2M_{wm_mode}_{mechanism}_{mean_str}_{std_str}_{model_name}/watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"

# Define update model function with new embeddings
def update_model(eps, model, mechanism, mean, std,model_name):
    device = model.device
    save_dir = 'emb/'
    if eps != 0:
        file_name = 'emb_' + mechanism + '_' + str(mean) + '_' + str(std) + '_c4_' + model_name + '_Noscaled.pt'
        file_path = os.path.join(save_dir,file_name)
        new = torch.load(file_path)
        new = torch.squeeze(new)
        newEmbed = nn.Embedding(model.config.vocab_size, model.config.hidden_size, _weight=new.to(device))
        model.set_input_embeddings(newEmbed)
        print('New embeddings added to the model')

# We load device map 
def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def main():
    args = parse_args()
    mode = args.mode
    wm_mode = args.wm_mode
    mechanism = args.mechanism
    mean = args.mean
    std = args.std
    model_name = args.model_name
    num_epochs = args.num_epochs
    learning_rate_ = args.learning_rate
    type_of_text = args.type_of_text
    num_training_data = args.num_training_data
    eps = args.eps
    # Print the parameters
    print('----------------Mode of finetune: ', args.mode)
    print('----------------Epsilon: ', args.eps)
    print('----------------Mean: ', args.mean)
    print('----------------Noise Scale: ', args.std)
    print('----------------Mechanism of Noise: ', args.mechanism)
    print('----------------Model: ', args.model_name)
    print('----------------Watermark type: ', args.wm_mode)
    print('----------------Type of text: ', args.type_of_text)
    print('----------------Number of training data: ', args.num_training_data)

    ##################
    start_time = time.time()
    # Clear cache
    torch.cuda.empty_cache()
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('----------------Device: ', device)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the adapter paths and load data
    train_file, test_file = get_file_paths(mode, wm_mode, model_name)
    train_data = load_data(train_file, type_of_text)
    train_data = train_data[:num_training_data]
    test_data = load_data(test_file, type_of_text)

    train_dataset = create_dataset(train_data)
    test_dataset = create_dataset(test_data)

    # Create the new model / adapter path
    new_model = get_new_model_path(mode, wm_mode, mechanism, mean, std, model_name, num_epochs, learning_rate_, type_of_text, num_training_data)
    # Create the directory of adapter if it doesn't exist
    if not os.path.exists(new_model):
        os.makedirs(new_model)

    # Load parameters
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set device map  
    if script_args.use_multi_gpu:
        device_map = "auto"
    else:
        device_map = {"":get_current_device()}

    # Check if 8 bit and 4 bit precision are used at the same time
    if script_args.use_8_bit and script_args.use_4_bit:
        raise ValueError(
            "You can't use 8 bit and 4 bit precision at the same time"
        )

    # Check if 4 bit precision is used and specify the Configration of QLoRA
    if script_args.use_4_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=script_args.use_bnb_nested_quant,
        )   
    else:
        bnb_config = None
    
    # Load the model and add the Quantized LoRA
    if model_name == 'llama2':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                                cache_dir=cache_dir,
                                                quantization_config=bnb_config, # add Quantized LoRA
                                                device_map={"": 0}
                                                )
    elif model_name == 'mistral7b':
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
                                                cache_dir=cache_dir,
                                                quantization_config=bnb_config, # add Quantized LoRA
                                                device_map={"": 0}
                                                )

    model.config.use_cache = False # Disables caching during training.
    model.config.pretraining_tp = 1 # Use Pretrained model

    # IMPORTANT: Update model with new embeddings
    update_model(eps, model, mechanism, mean, std, model_name)

    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model) 
    
    # Load tokenizer
    if model_name == 'llama2':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif model_name == 'mistral7b':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=False)

    # Add special tokens
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Config   
    peft_config = LoraConfig(
            lora_alpha=32, # Alpha value for LoRA, the higher the value, the more aggressive the sparsity
            lora_dropout=0.05,
            r=16, # Rank of the LoRA decomposition
            target_modules= ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'], # Target modules to apply LoRA
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Store loss for visualization
    class LoggingCallback(TrainerCallback):                
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                output_log_file = os.path.join(args.output_dir, "train_results.json")
                with open(output_log_file, "a") as writer:
                    writer.write(json.dumps(logs) + "\n")

    # Training arguments
    training_arguments = TrainingArguments(
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_steps=-1,
        save_total_limit=1, # Saving only 1 to save storage.
        logging_steps=500,
        eval_steps=500,
        learning_rate=learning_rate_, 
        weight_decay=0.001,
        per_device_train_batch_size=script_args.batch_size,
        max_steps=-1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=script_args.batch_size,
        output_dir=new_model,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        optim=script_args.optimizer_name,
        fp16=True,
        logging_strategy="steps",  # Ensure this is set to log at each step
        log_level='info'
    )
    
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        args=training_arguments,
        callbacks=[LoggingCallback()]  # Add the custom callback here
    )

    # Train the model
    trainer.train()
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)
    print('Done in ', time.time()- start_time)

    ##################
    # Save plots
    # Initialize lists to store the metrics
    epochs = []
    train_losses = []
    eval_losses = []

    # Load evaluation results
    eval_results_file = os.path.join(new_model, "train_results.json")
    with open(eval_results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if 'epoch' in data:
                epoch = data['epoch']
                if 'loss' in data:
                    train_losses.append(data['loss'])
                    epochs.append(epoch)  # Append epoch only when train loss is present
                if 'eval_loss' in data:
                    eval_losses.append(data['eval_loss'])
                    if epoch not in epochs:
                        epochs.append(epoch)  # Append epoch only when eval loss is present

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss', color='blue')
    plt.plot(epochs[:len(eval_losses)], eval_losses, label='Eval Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Evaluation Loss: {mode}_{wm_mode}_{mechanism}_{str(mean)}_{str(std)}_{model_name}_{str(num_epochs)}_lr{str(learning_rate_)}_{type_of_text}', fontsize=10)
    plt.legend()

    plt.tight_layout()

    # Save the plot in the current directory
    plot_path = os.path.join(new_model, 'training_evaluation_loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved in the current directory as 'training_evaluation_loss_plot.png'.")
    print('Done!')


if __name__ == "__main__":
    main()





