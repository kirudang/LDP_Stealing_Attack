
# SafeSeal: Certifiable Watermarking for LLM Deployments
## Requirements
To facilitate the setup, we recommend creating a seperate environment and installing the necessary packages from `safeseal.yml`. The simulations were conducted on Python version 3.9.20,using NVIDIA A100 GPU with PyTorch (torch 2.5.1) and CUDA 12.
```bash
conda env create --file safeseal.yml
```
## I. SafeSeal Watermarking
All code related to our paper is located in the `src/` folder, and a sample text in json file in `data/` folder. Instructions are as below: 

### 1. Apply Watermark to the Original Text
Folder `Generation/` contains the code for watermark generation. 

To apply the watermark to the original non-watermarked text, you need to run the `SafeSeal_Generation.py` script. 

Parameters to modify:
- `os.environ["HF_TOKEN"]` = 'Your_HuggingFace_Token' # Hugging Face token for model access
- `cache_dir` = 'Your/Cache/Directory' # Directory for model caching
- `N_start` = 0 # Start index for the text to be watermarked
- `N_end` = 100 # End index for the text to be watermarked
- `output_name` = f"Train_Llama3_top_{Final_K}_threshold_{threshold}_Uniform_{N_start}_{N_end}" # Name of the output file
- `data` (In parser): Data in json format to apply watermarking
- `final_k` (In parser): Final K alternatives to consider
- `threshold` (In parser): Threshold for similarity. Our default is 0.8, but you can adjust it based on your needs. 

For other files, you may need to update the Hugging Face token and cache directory.
In order to expidite the process, we also use caching in `utils.py`. For different setting, you should update `CACHE_FILE = "lookup_cache_llama3.json`.

**Note**: Our machenism uses BLEURT as the similarity metric for candidate selection. You can download the BLEURT model (Google research team) from [here](https://github.com/google-research/bleurt). The model will be cached in the `cache_dir` specified in `BLEURT_scores.py`.

---
In case you want to try other similarity metrics, you can use alternative metrics such as BERTScore (`BERT_scores.py`), and Setence-BERT (`ST_scores.py`), and modify refered code from `utils.py` file.


### 2. Watermark detection
Folder `Detection/` contains the code for watermark detection. The watermark detection process is divided into three steps: generating watermarked text, training the watermark detector, and detecting the watermark.
#### Step 1: Generate Watermarked Text
You need to generate watermarked text for training our Watermark detector using the `SafeSeal_Generation.py` script above. 


#### Step 2: Train the Watermark Detector

To train the watermark detector, you need to run the `IPC_train.py` script. Modify the following parameters:
- `lrate` = 1e-5 # Suggest to keep it as is
- `epoches_num` = 10 # Suggest to keep it as is
- `batch_size` = 32 # Suggest to keep it as is
- `base_model` = "microsoft/deberta-base-mnli" # Suggest to keep it as is
- `short_name` = "Deberta" # Suggest to keep it as is
- `test_file` = "Yourfile.json", with `Watermarked_output` and `Original_output` columns.
- `train_file` = "Yourfile.json", with `Watermarked_output` and `Original_output` columns.

#### Step 3: Detect Watermark
To detect the watermark, you need to run the `IPC_test.py` script. Modify the following parameters:
- `IPC_dir` = ""  # The directory where the trained model is saved in Step 2.
- `input_json` = "File_to_test.json"  # Path to the input JSON file for evaluation
- `column` ="Watermarked_output" # Column name from input file to evaluate
- `output_csv` = "Detection_LLama.csv"  # Path to save the predictions CSV

**Note:** Watermarked text will be return as 1 and non-watermark text as 0.

## II. Other Watermark Implementation

We adhere to the original settings specified in their uploaded codes, allowing for straightforward replication. Please refer to the detailed guidance provided for each type of watermark by accessing the following resources:
- KGW: [KGW](https://github.com/jwkirchenbauer/lm-watermarking)
- EXP: [EXP](https://github.com/jthickstun/watermark)
- SIR: [SIR](https://github.com/THU-BPM/Robust_Watermark)
- SynthID: [SynthID](https://github.com/google-deepmind/synthid-text)
- DTM: [DeepTextMark](https://github.com/tanvir097/DeepTextMark)
- TW: [TW](https://github.com/Kiode/Text_Watermark)


## III. Other experiments - Watermark Removal Attacks
To compare our attack with others, we used the following settings:
- Dipper: [ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases/tree/main) (Parameters: `lex = 60`, `order = 60`)
- Substitution Attack: [text_editor](https://github.com/THU-BPM/MarkLLM/blob/main/evaluation/tools/text_editor.py) (Parameter: `ratio = 0.7`).

## IV. Other experiments - Evaluation Metrics
To evaluate the performance of our watermarking method, we used the following metrics in folder `Evaluation_Metrics`:
- BERTScore: `BERTscore_Eval.py`
- Entity Similarity Score: `Entity_Similarity_Eval.py`


Enjoy the code!
