import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers import BitsAndBytesConfig
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Nethook
import llava.eval.nethook_multimodal as nethook_mult

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import random 
import numpy as np 
import json 
import pickle 


print(f'Loaded')
device = 'cuda'  # or cpu
torch.set_default_device(device)

# Function to get the relevant layer names 
def get_layer_names(model):
    # 
    layer_names = []
    for n, m in model.named_modules():
        if n.split('.')[-1] == 'self_attn' or n.split('.')[-1] == 'mlp':
            if 'vision_model' not in n:
                layer_names.append(n)

    return sorted(layer_names)

# Set Seed function
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# Decode Tokens
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


# Finding Token Range
def find_token_range(tokenizer, input_ids, prompt, current_constraint):
    #print(input_ids.shape)
    #print(f'Finding the token range ....')
    input_ids_curr = torch.cat([input_ids[0][0:1], input_ids[0][2:]])
    toks = [tokenizer.decode([t]) for t in input_ids_curr]

    # Whole string
    whole_string = "".join(toks)
    substring = "".join(current_constraint.split(" "))
    char_loc = whole_string.index(substring)
    print(toks)
    #print(f'Character localization : {char_loc}')
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break

    # Return the 
    return (tok_start, tok_end)


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
     
    #raise LookupError(name)
    return None



# Evaluation # 
def eval_model(args):
    # Model
    #disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    ######## Load Llava ########
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    noise_level = args.noise_level 
    uniform_noise = False

    print(f'Device of the model : {model.device}')


    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            #print(f'Factor of noise : {factor}')
            # noise_level = factor * collect_embedding_std(
            #     mt, [k["subject"] for k in knowns]
            # )
            noise_level = factor * 0.13
        
    
    # Define the answer to the question
    answer = args.trace_answer #'Seattle'
    # Define the question
    question = args.trace_question #'This building is located in?' 
    # Retrieve the image 
    image = Image.open(args.trace_image).convert('RGB')
    # Define the constraint
    current_constraint = args.trace_constraint
    # Define the prompt required for the question 
    prompt = "<image>\nUSER: " + question + "? Answer in a few words.\nASSISTANT:"

    
    differences, probs_clean, probs_corr = calculate_hidden_flow(args, image_processor, tokenizer, model, noise_level, image, prompt, answer, current_constraint, question)

    scores = []
    layers_store = []
    for sc in differences:
        scores.append(sc[2].item())
        layers_store.append([sc[0], sc[1]])
    
    # Scores # 
    scores = np.array(scores)
    layers_store = np.array(layers_store)

    # 
    indexes = np.argsort(scores)[::-1]
    print(f'# Sorted Scores: {scores[indexes][:10]}')
    print(f'# Layers: {layers_store[indexes][:10]}')

    



# Function to compute the clean and corrupted model ---> Run multimodal causal trace # 
def calculate_hidden_flow(args, image_processor, tokenizer, model, noise_level, image, prompt, answer_, current_constraint, question):
    # Original input_ids
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device) #cuda(0)
    input_ids_2 = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device) #cuda(0)
    # # Image Tensor
    image_tensor = process_images([image], image_processor, model.config)[0].to(dtype=model.dtype, device=device)
    image_tensor_2 = process_images([image], image_processor, model.config)[0].to(dtype=model.dtype, device=device)
    
    input_ids_final = torch.cat([input_ids, input_ids_2, input_ids_2]) #input_ids.expand(3,-1)
    image_tensor_final = torch.stack((image_tensor, image_tensor_2, image_tensor_2))#image_tensor.expand(3, 3, 336, 336)

  
    print(f'# Step 1: Create the clean model # ')
    ############ Clean model ##############
    with torch.no_grad():
        out = model(
            input_ids_final,
            images=image_tensor_final.half().cuda(0),
            image_sizes=[image.size],
            use_cache=True)["logits"]

    # Probabilites
    probs = torch.softmax(out[:, -1], dim=1)
    
    # Prediction
    p, preds = torch.max(probs, dim=1) # Computing P(O)

    print(f'Highest Probability : {p}')
    print(f'Token : {preds}')
    answer = tokenizer.decode(preds[0])
    print(f'# Answer :{answer} # ')

    # 
    print(f'# Step 2: Create the corrupted model # ')
    e_range = find_token_range(tokenizer, input_ids, prompt, current_constraint)
    print(f'E-range: {e_range}')
    # Low-score # 
    low_score = trace_with_repatch(args, model,tokenizer, e_range, input_ids_final, image_tensor_final, image.size, preds[0].item())
    probs_clean = low_score[0]
    probs_corrupt = low_score[1]
    
    print(f'Probs Clean: {probs_clean}')
    print(f'Probs Corrupt: {probs_corrupt}')

    # Difference # 
    difference = (probs_clean - probs_corrupt) / probs_clean

    # Difference in the probability # 
    print(f'# Difference # : {difference}')

    layer_names = get_layer_names(model)
    token_range = None 
    differences = trace_important_states(
            args,
            model,
            preds[0].item(),
            tokenizer, 
            layer_names,
            input_ids_final,
            e_range,
            image.size,
            image_tensor_final, 
            noise=None,
            uniform_noise=None,
            replace=None,
            token_range=token_range,
    )




    return differences, probs_clean, probs_corrupt 

# Function to trace the important states 
def trace_important_states(
            args,
            model,
            token_answer,
            tokenizer, 
            layer_names,
            input_ids_final,
            e_range,
            image_size,
            image_tensor_final,
            noise,
            uniform_noise,
            replace,
            token_range,
    ):

    # 
    print(f'################# Tracing Important States ##################')
    token_range = list(range(576, 576 + len(input_ids_final[0])-1))

    # Results #
    results = []

    # Iterate through the token range
    for tok_idx in token_range:
        # Iterate through layer names
        c = 0
        for l in layer_names:
            #print(f'Token Index : {tok_idx} ====> Layer :{l}')
            if args.window == 1:
                curr_layer_list = []
                #curr_layer_list.append([tok_idx, 'model.mm_projector'])
                curr_layer_list.append([tok_idx, l])
                
                with torch.no_grad(), nethook_mult.TraceDict(model, curr_layer_list, retain_output=False, edit_output='single') as t: # Edit-output option ----> single/window
                    # Perform the forward pass ---> 
                    out = model(
                        input_ids_final,
                        images=image_tensor_final.half().cuda(0),
                        image_sizes=[image_size],
                        use_cache=True)["logits"]

                    # 
                    probs = torch.softmax(out[:, -1], dim=1)
                    probs_clean = probs[0][token_answer]
                    probs_corrupted = probs[1][token_answer]
                    probs_restored = probs[2][token_answer]


                    results.append([tok_idx, l, probs_restored])
            

            else:
                curr_layer_list = []
                curr_layer_list.append([tok_idx, 'model.mm_projector'])
                if 'mlp' in l:
                    # MLP layer
                    curr_index = int(l.split('.')[2])
                    temp_layers = []
                    start_index = max(curr_index-args.window, 0)
                    end_index = min(curr_index + args.window,31)

                    for k in range(start_index, end_index+1):
                        curr_layer_list.append([tok_idx, "model.layers." + str(k) + ".mlp"])

                    

                elif 'self_attn' in l:
                    # Self-attention layer
                    curr_index = int(l.split('.')[2])
                    temp_layers = []
                    start_index = max(curr_index-args.window, 0)
                    end_index = min(curr_index + args.window,31)

                    for k in range(start_index, end_index+1):
                        curr_layer_list.append([tok_idx, "model.layers." + str(k) + ".self_attn"])


                # torch.no_grad()
                with torch.no_grad(), nethook_mult.TraceDict(model, curr_layer_list, retain_output=False, edit_output='single') as t: # Edit-output option ----> single/window
                    # Perform the forward pass ---> 
                    out = model(
                        input_ids_final,
                        images=image_tensor_final.half().cuda(0),
                        image_sizes=[image_size],
                        use_cache=True)["logits"]

                    # Probabilities
                    probs = torch.softmax(out[:, -1], dim=1)
                    probs_clean = probs[0][token_answer]
                    probs_corrupted = probs[1][token_answer]
                    probs_restored = probs[2][token_answer]

                    # Append the results
                    results.append([tok_idx, l, probs_restored])
            
        
                    
            


    return results 

# Function to obtain the corrupted model
def trace_with_repatch(args, model, tokenizer, e_range, input_ids_final, image_tensor_final, image_size, prediction_token):
    print(f'# Creating the corrupted model.')
    token_to_add = args.corrupt_token
    token_id = tokenizer.encode(token_to_add)
    

    # Number of tokens to replace # 
    difference = e_range[1] - e_range[0]
    
    for k in range(0, difference):
        input_ids_final[1][e_range[0] + k + 1] = token_id[1]
        input_ids_final[2][e_range[0] + k + 1] = token_id[1]

    
    with torch.no_grad():
        out = model(
            input_ids_final,
            images=image_tensor_final.half().cuda(0),
            image_sizes=[image_size],
            use_cache=True)["logits"]

        # Probabilites
        probs = torch.softmax(out[:, -1], dim=1)
        
        #print(probs.shape)
        probs_clean = probs[0][prediction_token]
        probs_corrupted = probs[1][prediction_token]

        p, preds = torch.max(probs, dim=1) # Computing P(O)
        # Answer with one forward pass # 
        answer = tokenizer.decode(preds[0])


    return probs_clean, probs_corrupted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--dataset', type=str, default='known', help='known')      # option that takes a value
    parser.add_argument('--model', type=str, default='llava', help='llava / bunny_phi / bunny_qwen')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--severed', type=str, default='no', help ='yes/no')
    parser.add_argument('--severed_type', type=str, default='mlp', help ='mlp/attn')
    parser.add_argument('--noise_level', type=str, default='s3', help ='s1/s2/s3')
    parser.add_argument('--replace', type=str, default='False', help ='True/False') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--prompt', type=str, default='None', help ='Any prompt') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--prompt_image_path', type=str, default='None', help ='Image path corresponding to the prompt') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--constraints', type=str, default='None', help ='Constraints corresponding to the prompts') # True for additive perturbations and False for replacement perturbations

    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--perform_trace", type=str, default="true")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2) # Default was 0.2
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--corrupt_token", type=str, default='Paris')

    parser.add_argument("--trace_answer", type=str, default='Seattle')
    parser.add_argument("--trace_question", type=str, default='This building is located in?')
    parser.add_argument("--trace_image", type=str, default='Path of image')
    parser.add_argument("--trace_constraint", type=str, default='This building')


    args = parser.parse_args()

    eval_model(args)
