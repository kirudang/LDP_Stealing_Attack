import os
import argparse
import torch
from transformers import AutoModelForCausalLM
import numpy as np

os.environ["HF_TOKEN"] = "Your_Huggingface_Token"
# Specify the directory path 
cache_dir = "Your_Cache_Directory"
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = cache_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Add noise to embeddings")
    
    parser.add_argument('--m1', type=int, default=0, help='Number of bits for the integer part in binary conversion')
    parser.add_argument('--m2', type=int, default=30, help='Number of bits for the fractional part in binary conversion')
    parser.add_argument('--eps', type=float, default=10, help='Small positive number for the irreducible function')
    parser.add_argument('--alg', type=str, default='lap', choices=['fLDP', 'gauss', 'lap'], help='Algorithm to use for adding noise, default is lap: Laplace')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for the noise')
    ### Change below parameters for your experiment
    parser.add_argument('--model_name', type=str, default='llama2', help='Name of the model to use for generating embeddings')
    parser.add_argument('--scale', type=float, default=0.01, help='Noise scale for the attack') # Important parameter for the attack
    
    return parser.parse_args()

def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.
    """
    x_bin = ''
    count = 0
    for i in range(len(x)):
        x_scaled = int(np.round(x[i] * 2 ** n))
        if (count == 0):
          count += 1
        if x_scaled > 0:
            k = '1' + '{:0{}b}'.format(x_scaled, m + n)
        else:
            k = '0' + '{:0{}b}'.format(-x_scaled, m + n)
        x_bin = x_bin + k
    return x_bin

def get_irr_funct(x, alpha, eps, l):
    """
    Generates an irreducible function based on the input parameters.

    Parameters:
    x (str): A binary string input.
    alpha (float): A scaling factor.
    eps (float): A small positive number.
    l (int): A length parameter.

    Returns:
    str: A binary string representing the irreducible function.
    """
    num_bits = len(x)
    irr = ''
    
    for i in range(num_bits):
        exp_term = np.exp(((i) % l) * eps / l)
        if x[i] == '1':
            prob = 1 / (1 + alpha * exp_term)
            bit = np.random.choice([0, 1], p=[1 - prob, prob])
        else:
            prob = (alpha * exp_term) / (1 + alpha * exp_term)
            bit = np.random.choice([0, 1], p=[1 - prob, prob])
        
        irr += str(bit)
    
    return irr

def binary_to_float(bstr, m, n):
    """
    Convert a binary string in the format given above to its float value.

    Parameters:
    bstr (str): The binary string to convert.
    m (int): The number of bits for the integer part.
    n (int): The number of bits for the fractional part.

    Returns:
    float: The converted float value.
    """
    if bstr[0] == '1':
        k = int(bstr[1:], 2) / 2 ** n
    else:
        k = -int(bstr[1:], 2) / 2 ** n

    return k


def gen_data(sentences, m1, m2, alpha, eps, l, alg, scale):
    """
    Generate data with added noise based on the specified algorithm.

    Parameters:
    sentences (list): List of sentences to process.
    m1 (int): Number of bits for the integer part in binary conversion.
    m2 (int): Number of bits for the fractional part in binary conversion.
    alpha (float): Scaling factor for the irreducible function.
    eps (float): Small positive number for the irreducible function.
    l (int): Length parameter for the irreducible function.
    alg (str): Algorithm to use for adding noise ('fLDP', 'gauss', 'lap').
    scale (float): Scale parameter for the noise.

    Returns:
    torch.Tensor: Tensor containing the processed data.
    """
    data = []

    for sent in sentences:
        emb = sent
        if alg == 'fLDP':
            binary = float_to_binary(emb, m1, m2)
            irr = get_irr_funct(binary, alpha, eps, l)
            irr_send = []
            for i in range(len(emb)):
                t = irr[i * (m1 + m2 + 1) : (i + 1) * (m1 + m2 + 1)]
                fl = binary_to_float(t, m1, m2)
                irr_send.append(fl)
        elif alg == 'gauss':
            irr_send = torch.tensor(emb) + torch.normal(mean=0, std=scale, size=emb.shape)
        elif alg == 'lap':
            dist = torch.distributions.laplace.Laplace(loc=0, scale=scale)
            noise = dist.sample(emb.shape)
            irr_send = torch.tensor(emb) + noise

        data.append(irr_send)

    data = np.stack(data, axis=0)
    data = torch.from_numpy(data)
    data = torch.squeeze(data)

    return data

def main():
    args = parse_args()
    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ##### LDP embeddings 
    m1 = args.m1
    m2 = args.m2
    len_ = m1 + m2 + 1 
    n = m1
    eps = args.eps
    alg = args.alg
    r = 768
    rl = r * len_
    mean = args.mean
    scale = args.scale


    if alg == 'fLDP':
        sum_ = 0
        for k in range(len_):
            sum_ += np.exp(2 * eps*k /len_)
        alpha = np.sqrt( (eps + rl) /( 2*r *sum_ )  ) 
    else: 
        alpha = 0

    model_name = "llama2"
    if model_name == 'llama2':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir=cache_dir)

    # Add noise to the embeddings
    data = torch.arange(32000) # Change this vocabulary size depend on the model, 32000 is for Llama2/ Mistral.
    emb = model.model.embed_tokens(data.long()).detach()
    #emb = model.model.decoder.embed_tokens(data.long()).detach() # Use this if the model is a decoder model
    print("Old embed: ", emb.size())

    emb = emb.detach().cpu().numpy() 
    print('emb------',emb)

    sData = gen_data(emb, m1, m2, alpha, eps, len_, alg, scale).to(dtype=torch.float32).unsqueeze(0)
    print("New embed: ", sData.size())
    
    save_path = 'emb/emb_{}_{}_{}_c4_{}_Noscaled.pt'.format(alg,mean,scale,model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sData, save_path) 
    print('sData------',sData)
    print(sData.shape)
    print('DONE!')

if __name__ == '__main__':
    main()


