## Understanding Information Storage and Transfer in Multimodal Language Models 

This is the implementation of the interpretability and model editing experiments from NeurIPS 2024 paper : https://arxiv.org/abs/2406.04236.


<img width="736" alt="Screen Shot 2024-09-23 at 5 09 12 PM" src="https://github.com/user-attachments/assets/4094fc67-5b41-4d93-8788-ef0d004f7e81">


================================================
### Constraint Annotations 

``` The constraint annotations are in ./data_constraints. The directory contains constraints for OK-VQA, Multimodal Known and Multimodal Movies. ```

================================================

### Images used for Probe Dataset 

For OK-VQA, we use the val set images from https://okvqa.allenai.org/. For Multimodal Known the images are at: [Link 1](https://drive.google.com/file/d/1YuGAomZdkMBvQBKndTimUFFmIWduahNS/view?usp=sharing) and for Multimodal Movies the images are at: [Link 2](https://drive.google.com/file/d/1n2mBeUyY7K3ZRpXHHH6fWXFWXnf2p0Sw/view?usp=sharing).

================================================

### Running the Scripts 

Our codebase is built on Llava's code. Clone [Llava](https://github.com/haotian-liu/LLaVA) and transfer the code from this repository to ```./llava/eval ```.

1. Multimodal Causal Trace: ``` python -m llava.eval.multimodal_trace --trace_answer <Answer for the prompt> --trace_question <Question> --trace_image <Image Path> --trace_constraint <Constraint in the question>```

2. Multimodal Edit: ``` python -m llava.eval.multimodal_edit --edit_prompt <Prompt used for running model editing> --edit_constraint <Constraints in th prompt> --edit_target <Target Answer> --edit_og_answer <Original Answer to the prompt> ```

================================================

#### Notes : Although the current script is optimized for Llava, these scripts can be modified towards applying it on any multimodal language model. Reach out to sbasu12@umd.edu for any questions.
