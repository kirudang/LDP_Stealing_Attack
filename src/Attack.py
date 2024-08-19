import os
# Set the HF_HOME environment variable to point to the desired cache location
os.environ["HF_TOKEN"] = "Your_Huggingface_Token"
# Specify the directory path 
cache_dir = 'Your_Cache_Directory'
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = cache_dir

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Attack from the finetuned model. Make sure to keep the parameters the same as the finetuned model.")
    ### Keep all parameters of the fine-tuned model to use the attack model.
    parser.add_argument('--mode', type=str, default='watermark', help="Mode of finetune: 'no_watermark' or 'watermark'")
    parser.add_argument('--mechanism', type=str, default='lap', help='Mechanism of noise: Laplace')
    parser.add_argument('--eps', type=float, default=10, help='Epsilon')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean')
    parser.add_argument('--type_of_text', type=str, default='input_output', help='Text column to use for finetuning in your dataset')
    parser.add_argument('--num_training_data', type=int, default=10000, help='Number of data to use for training, default is 10000')
    parser.add_argument('--model_name', type=str, default='llama2', help='Model to use for finetuning')
    parser.add_argument('--wm_mode', type=str, default='icml', help="Watermark type to attack on. The others are: 'rdf', 'SIR', 'semstamp'")
    parser.add_argument('--std', type=float, default=0.01, help='Noise scale. It is the --scale in Add_noise_embedding.py')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    ### Change below parameters for your attack experiments, which is relevant to watermark benchmark.
    parser.add_argument('--data_path', type=str, default='.../../../c4_promt_test.pt', help='The same data path of generating the watermarked text')
    parser.add_argument('--max_inp', type=int, default=200, help='Maximum input size. The same number of tokens used for generating the watermarked text')
    parser.add_argument('--max_out', type=int, default=200, help='Maximum output size. The same number of tokens used for generating the watermarked text')
    parser.add_argument('--Ninputs', type=int, default=2000, help='How many inputs to run the model on')
    parser.add_argument('--saving_freq', type=int, default=10, help='Frequency of saving the attack results')

    return parser.parse_args()

# Adapter path
def get_adapter_path(mode, wm_mode, mechanism, mean, std, model_name, num_epochs, learning_rate_, type_of_text, eps, num_training_data):
    mean_str = str(mean)
    std_str = str(std)
    num_epochs_str = str(num_epochs)
    learning_rate_str = str(learning_rate_)
    num_training_data_str = str(num_training_data)

    if eps == 0:
        base_path = f"./Finetune_newmodels/noLDP/save_noS2M_{wm_mode}_"
        if mode == 'watermark':
            return f"{base_path}watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"
        elif mode == 'no_watermark':
            return f"{base_path}no_watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"
    else:
        base_path = f"./Finetune_newmodels/save_noS2M_{wm_mode}_{mechanism}_{mean_str}_{std_str}_{model_name}"
        if mode == 'watermark':
            return f"{base_path}/watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"
        elif mode == 'no_watermark':
            return f"{base_path}/no_watermark_{num_epochs_str}_lr{learning_rate_str}_{type_of_text}_{num_training_data_str}"


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
    print('----------------Attack Model Information:-------------')
    print('----------------Mode of finetune: ', args.mode)
    print('----------------Epsilon: ', args.eps)
    print('----------------Mean: ', args.mean)
    print('----------------Noise Scale: ', args.std)
    print('----------------Mechanism of Noise: ', args.mechanism)
    print('----------------Model: ', args.model_name)
    print('----------------Watermark type: ', args.wm_mode)
    print('----------------Type of text: ', args.type_of_text)
    print('----------------Number of training data: ', args.num_training_data)
    
    ################## START OF THE MAIN FUNCTION ##################
    start_time = time.time()
    # Clear cache
    torch.cuda.empty_cache()
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('----------------Device: ', device)

    # Base model
    if model_name == 'llama2':
        LLM_name = "meta-llama/Llama-2-7b-chat-hf"
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0})
    elif model_name == 'mistral7b':
        LLM_name = "mistralai/Mistral-7B-Instruct-v0.2"
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0})

    # Get Adapter Path from Finetune
    adapter = get_adapter_path(mode, wm_mode, mechanism, mean, std, model_name, num_epochs, learning_rate_, type_of_text, eps, num_training_data)

    # Check if the adapter path exists
    print(f'Path to adapter: {adapter}')
    if os.path.exists(adapter):
        print("Path exists.")
    else:
        print("Path does not exist. Exiting...")
        exit()

    # Merge the base model and adapter for attack
    model = PeftModel.from_pretrained(base_model, adapter)
    model = model.merge_and_unload()
    model.to(device)

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM_name, cache_dir=cache_dir,use_fast=False)
    # Add special tokens
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token

    # Load the data
    data = torch.load(args.data_path)

    # Initialize an empty list to store generated outputs
    generated_data = []
    input_counter = 0

    # Run the finetuned model on the data to generate the attack results
    for text in data[0][:args.Ninputs]:
        # Encode the first 200 tokens of each text
        prompt_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.max_inp).to(device)
        prompt_tokens = prompt_tokens['input_ids'].cuda()
        generated_prompt = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=True)[0]

        # Generate the next 200 tokens
        outputs = model.generate(
            prompt_tokens,
            max_length=args.max_inp+args.max_out,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_output_text = tokenizer.decode(outputs[0][args.max_inp:args.max_inp+args.max_out], skip_special_tokens=True)

        # Store the input and output in a dictionary
        data_dict = {
            "input": generated_prompt,  # Store only the first 200 tokens of input
            "full_paraphrased_response": generated_text,
            "paraphrased_response": generated_output_text
        }

        # Append the dictionary to the list of generated data
        generated_data.append(data_dict)

        # Increment input counter
        input_counter += 1

        # Save the results after processing every 10 inputs
        if input_counter % args.saving_freq == 0:
            # Check if the file exits
            if os.path.isfile("output_tuned_" + type_of_text + "_" + model_name + "_" + mode + "_" + wm_mode + "_" + mechanism + "_" + str(std) + "_with_" + str(num_epochs) + "_training_data_" + str(num_training_data) + "_" + str(input_counter-args.saving_freq) + ".json"):
                os.remove("output_tuned_" + type_of_text + "_" + model_name + "_" + mode + "_" + wm_mode + "_" + mechanism + "_" + str(std) + "_with_" + str(num_epochs) + "_training_data_" + str(num_training_data) + "_" + str(input_counter-args.saving_freq) + ".json")

            with open("output_tuned_" + type_of_text + "_" + model_name + "_" + mode + "_" + wm_mode + "_" + mechanism + "_" + str(std) + "_with_" + str(num_epochs) + "_training_data_" + str(num_training_data) + "_" + str(input_counter) +  ".json", "w") as json_file:
                json.dump(generated_data, json_file, indent=4)


    # End time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time} seconds")
    print("DONE!")

if __name__ == "__main__":
    main()
