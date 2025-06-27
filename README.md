
# 𝛿-STEAL: LLM Stealing with Local Differential Privacy
## Requirements
To facilitate the setup, we recommend creating a dedicated environment and installing the necessary packages from `stealing.yml`. The simulations were conducted on Python version 3.10.13,using NVIDIA A100 GPU with PyTorch (torch 2.4.0) and CUDA 12.
```bash
conda env create --file stealing.yml
```
## Experiments with our 𝛿-STEAL Attack
All code related to our paper is located in the `src/` folder, and datasets are stored in the `data/` folder. Instructions for reproducing the results or training the model with your dataset are provided.
### Step 1: Add Noise to the Model Embeddings
We applied Laplace noise in our experiments. To execute the attack, modify the following parameters in `Add_noise_embedding.py`:
-  `model_name`: Specify the model of the original texts that you want to attack (default: Llama2).
-  `scale`: Set the noise scale for the attack (default: 0.01; options include 0.05 or 0.1 for enhanced efficacy).
To run this script:
```bash
python Add_noise_embedding.py
```
**Important Notes:**
- Adjust `data = torch.arange(32000)` to match the vocabulary size of your model (32000 is for Llama2/Mistral).
- If your model is an encoder, replace `emb = model.model.embed_tokens(data.long()).detach()` with `emb = model.model.decoder.embed_tokens(data.long()).detach()`.
### Step 2: Fine-Tune the Attack Model on Watermarked Text with Noise, using LoRA
To proceed with this step, ensure you have the necessary training and testing data prepared with the watermark scheme targeted for the attack. Detailed instructions on generating watermarked texts using the data in the `data/` folder are provided below. 
During this session, we continue to use LoRA for fine-tuning. However, a key adjustment involves updating the model with new embeddings via the `update_model` function.
```bash
# IMPORTANT: Update model with new embeddings
update_model(eps, model, mechanism, mean, std, model_name)
```
To execute this script:
```bash
python Finetune_attack_model_with_LoRA.py
```
### Step 3: Execute the Fine-Tuned Attack Model on Watermarked Text with Noise
This step involves deploying the recently fine-tuned model to execute an attack on the watermarked text. Unlike conventional methods such as paraphrasing or substituting, our approach generates attack texts directly from the original prompt used for watermark generation. Therefore, it is crucial to use our attack model while maintaining the same settings that were used during the watermark generation. For instance, retain 200 tokens from the `c4_promt_test.pt` dataset as the prompt, and generate the attack output for the subsequent 200 tokens.
To execute this script:
```bash
python Attack.py
```
## Other experiments - Watermark generation
This part instructs you how to generate watermark texts to feed the attack model.
### Data
- Training data: `c4_promt_test.pt` from the `data/` folder
- Testing data: `c4_promt_train.pt` from the `data/` folder

### Watermark Implementation and Detection

We adhere to the original settings specified in their uploaded codes, allowing for straightforward replication. Please refer to the detailed guidance provided for each type of watermark by accessing the following resources:
- KGW: [KGW](https://github.com/jwkirchenbauer/lm-watermarking)
- EXP: [EXP](https://github.com/jthickstun/watermark)
- SIR: [SIR](https://github.com/THU-BPM/Robust_Watermark)
- Semstamp: [SemStamp](https://github.com/bohanhou14/SemStamp)

## Other experiments - Benchmarking Other Watermark Attacks
To compare our attack with others, we used the following settings:
- Dipper: [ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases/tree/main) (Parameters: `lex = 60`, `order = 60`)
- Substitution Attack: [text_editor](https://github.com/THU-BPM/MarkLLM/blob/main/evaluation/tools/text_editor.py) (Parameter: `ratio = 0.7`).
- Watermark Removal: [paraphrasing_attack](https://github.com/hlzhang109/impossibility-watermark)

Enjoy the code!
