# Responsible AI FAQ: Understanding Knowledge Storage and Transfer in Multimodal Language Models

## What is the project about?

Our aim is to understand the inner workings of Multimodal Language Models (MLLM), such as LLava and Phi-vision. To do this, we investigate how MLLMs process information for a representative VQA task. This work has three parts:

1. The curation of a VQA dataset with the addition of constraints.
2. A novel tracing algorithm for identifying the layers in the model from which information is achieved for a VQA task.
3. A validation approach that incorporates rare data into the model for testing purposes.

This project has been accepted as a full paper in NeurIPS 2024: [https://arxiv.org/abs/2406.04236v1](https://arxiv.org/abs/2406.04236v1)

## What can the methods in the project do?

- **MultimodalCausalTrace** is a novel tracing algorithm that identifies the important layers from where information is retrieved in a MLLM for solving a VQA task.
- **MultEdit** validates the existence of these layers by incorporating rare knowledge into the model by editing the identified layer(s). This validation is performed at inference time per query, hence the editing is very specific to a query and does not affect any other queries, nor does it create a new edited checkpoint/model.

## What is in the dataset?

We introduce a dataset **VQA-Constraints** described in the paper [https://arxiv.org/abs/2406.04236v1](https://arxiv.org/abs/2406.04236v1). It is built from the following existing open-source datasets:

- **OK-VQA**: [https://okvqa.allenai.org](https://okvqa.allenai.org)
- **Known**: [https://arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)
- **Movies**: [https://arxiv.org/abs/2309.15098](https://arxiv.org/abs/2309.15098)

These datasets are then annotated for constraints. For example, the constraint for the question: “Where is this building located” is “this building.” We annotate such constraints for 9.7k questions.

## What is/are the project’s intended use(s)?

The intended use of the project is to mechanistic
ally understand the inner workings of Multimodal Language Models. We envision our methods being used by the research community to further the understanding of large foundational vision-language models.

## How were the methods evaluated? What metrics are used to measure performance?

To evaluate the tracing mechanism which identifies a set of important layers for a VQA task, we edit these layers to incorporate rare knowledge. We compute the VQA-Accuracy as a metric to capture the effectiveness of the edits to the identified layers. Consider an image-question pair (I,Q) with the ground-truth answer to be A. The model, however, gives a wrong answer A’. Post-edit, VQA-accuracy checks if the model is able to answer correctly with the generated answer as the original ground-truth answer A. Please refer to Fig. (6) in the paper for the evaluation results.

## What are the limitations of the project? How can users minimise the impact of the project’s limitations when using the system?

The project primarily investigates the inner workings of Multimodal Language Models for a representative VQA task. The primary limitation is that the insights in our paper are restricted to a factual VQA task. The results in the paper are not generalisable for a different vision-language task (e.g., non-factual VQA, image captioning etc).

We have not tested our method and dataset for understanding knowledge storage across disparate cultural/geographical questions. It may exhibit the same biases, errors or omissions as any dataset sourced from the open-source community and used as standard benchmarks for VQA.

Our project was designed and tested in the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.

Our project was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

## What operational factors and settings allow for effective and responsible use of the project?

One can use our editing method at inference time for a very specific query. Therefore, the editing method does not create a globally modified model and hence any AI harms mitigations in the base Multimodal Language Model will remain intact.

- **Human involvement**: We recommend human oversight to review the specific queries on which the editing method is used.
- **LLMs**: Users can choose the LLM that is optimised for responsible use. We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and harms mitigations with the latest industry standards for responsible use. The default LLM is Vicuna (built on top of Llama) which inherits the existing harms mitigation mechanisms and filters from the LLM provider.
- **Content Safety**: Given that we are not modifying the model globally, we are not changing the original content safety of the model and it remains similar to the base model.
