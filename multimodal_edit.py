""" 
Implementation of MultEdit : Editing Multimodal Language Models in early layers

"""

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
import nethook_llava

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
from torch.distributions import Categorical
# Matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import norm
import matplotlib.mlab as mlab
from sklearn.metrics import roc_auc_score
from misc_utils import repeat_kv, find_within_text
import edit_hook
import copy 

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


# Function to get the relevant layer names 
def get_layer_names(model):
    # 
    layer_names = []
    for n, m in model.named_modules():
        if n.split('.')[-1] == 'self_attn' or n.split('.')[-1] == 'mlp':
            if 'vision_model' not in n:
                layer_names.append(n)

    return sorted(layer_names)

# Get the module
def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        # 
        if n == name:
            #print(f'## Name : {n}')
            return m
    raise LookupError(name)


# Finding Token Range
def find_token_range(tokenizer, input_ids, prompt, current_constraint):
    #print(input_ids.shape)
    print(f'Finding the token range ....')
    input_ids_curr = torch.cat([input_ids[0][0:1], input_ids[0][2:]])
    
    toks = [tokenizer.decode([t]) for t in input_ids_curr]

    #print(toks)
    # Whole string
    whole_string = "".join(toks)
    substring = "".join(current_constraint.split(" "))
    char_loc = whole_string.index(substring)

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


# Function to invoke the main() function
# Main function 
def main():
    # TODO : Argparser
    # Define the argparser
    parser = argparse.ArgumentParser()
    
    # Argparser
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    epilog='Text at the bottom of help')
    
    # Arguments 
    parser.add_argument('--dataset', type=str, default='known_single', help='chartqa/okvqa_s/okvqa_m/attn_movies/attn_olympics/known/single')      # option that takes a value
    parser.add_argument('--model', type=str, default='llava', help='CoCa/BLIP-2/Open-Flamingo/LLaVa')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--severed', type=str, default='no', help ='yes/no')
    parser.add_argument('--severed_type', type=str, default='mlp', help ='mlp/attn')
    parser.add_argument('--noise_level', type=str, default='s3', help ='s1/s2/s3')
    parser.add_argument('--replace', type=str, default='False', help ='True/False') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--prompt', type=str, default='None', help ='Any prompt') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--prompt_image_path', type=str, default='None', help ='Image path corresponding to the prompt') # True for additive perturbations and False for replacement perturbations
    parser.add_argument('--constraints', type=str, default='None', help ='Constraints corresponding to the prompts') # True for additive perturbations and False for replacement perturbations

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--metric", type=str, default="attention_weights", help="Options: Attention weights, Attention contributions, Norm of embeddings in causal layers")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--grad_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1) # Learning rate #
    parser.add_argument("--weight_decay", type=float, default=0.001) # Weight decay #
    parser.add_argument("--regularization", type=float, default=0.01) # Weight decay #
    parser.add_argument("--layer_edit", type=int, default=2) # Weight decay #

    ################################# Editing parameters ###############################
    parser.add_argument("--imgpath", type=str, default='None') # Weight decay #
    parser.add_argument("--edit_prompt", type=str, default='None') # Weight decay #
    parser.add_argument("--edit_constraint", type=str, default='None') # Weight decay #
    parser.add_argument("--edit_target", type=str, default='None') # Weight decay #
    parser.add_argument("--edit_subject", type=str, default='None') # Weight decay #
    parser.add_argument("--edit_og_answer", type=str, default='None') # Weight decay #

    # Args
    args = parser.parse_args()

    # Model to use for editing # 
    if args.model == 'llava':
        set_seed(100)

        args.model_path = 'liuhaotian/llava-v1.5-7b'
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        print(f'## Model Path : {model_path} ## ')
        print(f'## Model Name : {model_name} ## ')
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, cache_dir='/data/models')
    

    # Requested Edit Operation #
    requested_edit_operation = {
        "image_path": args.imgpath, 
        "prompt": args.edit_prompt, 
        "subject": args.edit_subject, # Optional #  
        "target_edit": args.edit_target,
        "curr_constraints": args.edit_constraint, # Constraint in the prompt # 
        "original_answer": args.edit_og_answer
    }
    
    # Differences #
    counterfactual_differences = []
    
    # Average Probability of the token of interest # 
    edited_probabilities = []
    original_probabilities = []

    
    print(f'Requested Edit Operation : {requested_edit_operation}')
    
    ################# Define the layers to edit #################
    layers_to_edit = [args.layer_edit]
    # Down-Projection Layer is the one which needs to be edited # # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) # # First up_proj, then act_fn, then down_proj # 
    model_layers_to_edit = ['model.layers.' + str(l) + '.mlp.down_proj' for l in layers_to_edit]
    print(f'######### Layers to edit : {model_layers_to_edit} ############ ')
    #############################################################
    #############################################################

    """ Part 2: Obtain the keys corresponding to the last subject token of interest """
    # Current Image Path
    curr_image_path = requested_edit_operation['image_path']
    image_curr = Image.open(curr_image_path).convert('RGB')
    image_tensor = process_images([image_curr], image_processor, model.config)[0]
    prompt = "<image>\nUSER: " + requested_edit_operation["prompt"] + "? Answer in a few words.\nASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Range of 
    e_range = find_token_range(tokenizer, input_ids, prompt, requested_edit_operation["curr_constraints"])
    

    # Activation # 
    activation_keys = {}
    # Initialize the output dimension #
    output_val_dim = 4096
    def getActivation(name):
        # the hook signature
        def hook(m, input, output):
            #print(f'Type: {input[0].shape}')
            #print(f'Name of the module: {name}')
            # Save the input for the input #
            activation_keys[name] = input[0][0][576 + e_range[1] - 1].detach()
            #print(f'Shape : {activation[name].shape}')
            output_val_dim = output.shape[2]#[0]
            

        return hook
    
    # Iterate through the layer hooks # 
    layer_hooks = {l: None for l in model_layers_to_edit}
    for l in model_layers_to_edit:
        # Current Module #
        curr_module = get_module(model, l)
        # Layer Hooks #
        layer_hooks[l] = curr_module.register_forward_hook(getActivation(l))


    ###################### Obtain Keys ####################
    with torch.no_grad():
        out = model(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(0),
            image_sizes=[image_curr.size],
            use_cache=True)["logits"]
    
    # Obtaining the probabiltiies / logits
    probs = torch.softmax(out[:, -1], dim=1)
    # Prediction
    p, preds = torch.max(probs, dim=1) # Computing P(O)
    # Decoding the answer
    answer = tokenizer.decode(preds)
    print(f'############### Answer : {answer}; Token: {preds}; Probability of Token: {p} ###################')
    
    
    # Remove layer hooks #
    for l in model_layers_to_edit:
        layer_hooks[l].remove()

    ########################### At this step they keys corresponding to the last subject token in the text part is saved #################################
    # Obtain the values
    ############# Requires optimization of a value token #############
    edit_answer = requested_edit_operation["target_edit"] # Target edit answer #
    print(f'############### Edit Answer : {edit_answer} ###################')
    edit_ids = tokenizer.encode(edit_answer)[1:]
    edit_token_id_index = edit_ids[0] # First part #


    # Layerdimension
    # Objective: Optimize the output of the layers in model_layers_to_edit # 
    # Step 1: Define the objective for optimization (NLL) #
    # Step 2: Define the gradient operations (for optimizing te value inside for the output)
    # Flow of the operation: (i) Define the variable which is responsible the update (same shape as the output of the layer variable of W_{proj}); (ii) Whenever forward pass is called, use that value as the output; (iii) Define the loss accordingly
    delta = torch.zeros((output_val_dim,), requires_grad=True, device="cuda")

    target_init = None 

    # Edit Output Function #
    def edit_output_fn(cur_out, cur_layer):
        #nonlocal target_init
        nonlocal target_init 

        print(f'# Into the edit-output function #')
        if target_init == None:
            target_init = cur_out[0][576 + e_range[1] -1].detach().clone()
        

        # Assigning the delta variable to the output # 
        cur_out[0][576 + e_range[1] - 1] += delta 

        return cur_out


    # Optimizer # 
    opt = torch.optim.Adam([delta], lr=args.lr)
    # Setting the parameters for upgrade to false 
    edit_hook.set_requires_grad(False, model)
    
    # For tracking loss # 
    loss_track = []
    # For tracking probability # 
    prob_track = []

    # Iterate through the number of gradient steps #
    for it in range(args.grad_steps):
        # Removing the prior gradients from the optimizer states #
        opt.zero_grad()
        print(f'############ Iteration : {it} ####################')
        with edit_hook.TraceDict(
            module=model,
            layers=model_layers_to_edit,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            # Forward pass #
            out = model(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(0),
            image_sizes=[image_curr.size],
            use_cache=True)["logits"]

            # Compute the loss # 
            #log_probs = torch.log_softmax(out, dim=2)
            relevant_probs = torch.log_softmax(out[:, -1], dim=1)

            # Current token probability # 
            curr_prob = relevant_probs[0][edit_token_id_index]

            softmax_prob = torch.softmax(out[:, -1], dim=1)[0][edit_token_id_index]

            #print(torch.sum(relevant_probs))
            weight_decay_loss = args.weight_decay * (torch.norm(delta)/torch.norm(target_init)**2)

            # NLL Loss
            nll_loss = -relevant_probs[0][edit_token_id_index]

            # Total Loss #
            loss_total = nll_loss + weight_decay_loss 

            # Loss backward # 
            loss_total.backward()
            opt.step()

            # Norm
            max_norm = 0.0001 * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

            # Current Loss # 
            print(f'Current Loss : {loss_total.item()}')

            # Appending the # 
            loss_track.append(loss_total.item())
            prob_track.append(softmax_prob.item())


   
    # Delta which is detached #
    delta_detached = delta.detach().clone()
    
    # Target Value #
    target_value = target_init + delta_detached 
    key_value = activation_keys
    ####################################################
    print(f'####################### PRE-EDITING ######################')
    with torch.no_grad():
        out = model(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(0),
        image_sizes=[image_curr.size],
        use_cache=True)["logits"]

    # 
    probs = torch.softmax(out[:, -1], dim=1)

    probs_relevant = probs[0][edit_token_id_index]
    print(f'Probability of the edited token : {edit_token_id_index} is {probs_relevant}')

    original_probabilities.append(probs_relevant.item())

    # Prediction
    p, preds = torch.max(probs, dim=1) # Computing P(O)
    # Decoding the answer
    answer = tokenizer.decode(preds)

    print(f'Prob: {p} ; Prediction: {preds}; Answer: {answer}')
    print(f'Probability of the original token : {p}')
    p_orig = p.item()

    # Closed for update # 
    module_curr, weight_curr = perform_closed_form_update(args, model, key_value, target_value, requested_edit_operation, model_layers_to_edit)
    
    print(f'###################### POST EDITING #########################')

    # Forward pass #
    with torch.no_grad():
        out = model(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(0),
        image_sizes=[image_curr.size],
        use_cache=True)["logits"]

    # Compute the loss # 
    #log_probs = torch.log_softmax(out, dim=2)
    probs = torch.softmax(out[:, -1], dim=1)

    probs_relevant = probs[0][edit_token_id_index]
    probs_og = probs[0][preds.item()]
    print(f'Probability of the relevant token : {edit_token_id_index} is {probs_relevant}')
    print(f'Probability of the original token : {preds.item()} is {probs_og}')
    p_counterfact = probs_relevant
    
    print(f'######## Counterfactual Probability P(O*): {p_counterfact};; P(O-correct): {probs_og}########################;;; Efficacy Magnitude: {p_counterfact - probs_og}')

    # Edited Probabiltiies # 
    edited_probabilities.append(p_counterfact.item())

    # Prediction
    p, preds = torch.max(probs, dim=1) # Computing P(O)
    # Decoding the answer
    answer = tokenizer.decode(preds)

    # Reassigning the original weights #
    module_curr.weight = weight_curr 


# Function to perform closed form edit operation
""" 
Function which edits the W_{down_proj} layer to modify the weights and map from key_value to target_value
"""
def perform_closed_form_update(args, model, key_value, target_value, requested_edit_operation, model_layers_to_edit):
    print(f'###################### Performing Edit Operation #########################')
    
    
    # Define the matrices # 
    layer_to_edit = model_layers_to_edit[0]

    # Projection Matrix 
    projection_matrix = None 
    correct_module = None 
    for n,m in model.named_modules():
        if n == layer_to_edit:
            #print(f'### Current Module for the layer : {layer_to_edit} ###')
            projection_matrix = m.weight 
            correct_module = m 

    # Original Matrix # ----> Copy
    og_matrix = copy.deepcopy(projection_matrix.detach())
    
    # Keys # 
    keys = key_value[model_layers_to_edit[0]].reshape(1,-1)
    # Vals #
    values = target_value.reshape(1,-1)

    # Define the identity matrix # 
    identity_matrix = torch.eye(keys.shape[1]).to('cuda:0') # (11008 x 11008)

    keys = keys.float()
    # Left part of the output matrix
    output_matrix_left = torch.linalg.inv(torch.matmul(keys.T, keys) + args.regularization*identity_matrix)

    # Right part of the matrix
    output_matrix_right = torch.matmul(keys.T, values) + args.regularization * og_matrix.T
    
    # Updated Matrix #
    final_updated_matrix = torch.matmul(output_matrix_left, output_matrix_right).T.to(torch.float16)

    # Correct Module #
    correct_module.weight = torch.nn.Parameter(final_updated_matrix)
    

    return correct_module, projection_matrix



# Main function
if __name__ == "__main__":
    # Main function
    main()
