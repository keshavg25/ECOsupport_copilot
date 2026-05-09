---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:5000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: help me change purchase {{Order Number}}
  sentences:
  - We understand that you're seeking assistance in correcting your shipping address.
    Our team is here to help you with that. Please provide us with your current address
    and the correct address you'd like to update it to, and we will make the necessary
    changes for you. Let us know if you have any further questions or concerns.
  - We understand that you are looking to change an article from purchase number {{Order
    Number}}. Your satisfaction is our priority, and we are here to assist you. To
    better understand your request, could you please provide details about the article
    you would like to change and the desired replacement? This information will help
    us process your request accurately and ensure a seamless shopping experience for
    you.
  - I see your concern about losing the bills from {{Salutation}} {{Client Last Name}}.
    I'm here to help you locate them. Can you please provide me with any specific
    details such as date range, invoice number, or any other information that might
    help in finding the bills? Rest assured, I'll do my best to assist you in retrieving
    the necessary information.
- source_sentence: want help correcitng my shipping address
  sentences:
  - I'll do my best! I completely understand your need to check your bills from {{Person
    Name}}. Let's work together to find a solution. To start, could you please provide
    me with more information about the bill you're looking for? For example, do you
    have an invoice number or a date range that we can use as a reference? Once we
    have this information, we'll be able to assist you better.
  - Sure, I can help you with finding information about the early termination fees.
    To see the details regarding early termination fees, you can refer to the terms
    and conditions section of your account agreement. It should outline the fees and
    penalties associated with terminating your account before the agreed-upon term.
    If you have any specific questions or need further assistance, feel free to let
    me know!
  - I'll do my best! I'm here to assist you with updating your address. Please provide
    me with your current shipping address and the new address you want to change it
    to, and I'll guide you through the process.
- source_sentence: I have a problem correcting te shipping address
  sentences:
  - We're here to help to us for help with checking the invoice from {{Salutation}}
    {{Client Last Name}}. I understand how important it is for you to access the details
    of the invoice you're looking for. To assist you better, could you please provide
    me with additional information such as the invoice number, date range, or any
    other relevant details? With these details, I can quickly locate and provide you
    with the necessary information.
  - Certainly! I'll be happy to assist you in correcting your address. To do so, please
    provide me with the correct details, including the street name, house number,
    city, and postal code. Once I have the correct information, I will make sure to
    update your shipping address accordingly.
  - Honored to assist! I'm clued in that you do not know how to speak with a person.
    I apologize if the previous instructions weren't clear. To speak with a person,
    you can usually find a "Contact" or "Support" page on our website. You can also
    try reaching out to our customer service hotline at {{Customer Support Phone Number}}
    or sending us a message through our live chat service. Our team is here to help
    and will gladly assist you with any questions or concerns you may have.
- source_sentence: problem with canceling purchase {{Order Number}}
  sentences:
  - I'm sorry to hear that you have entered the wrong delivery address. I can help
    you with updating it. Please provide me with the correct address, and I will make
    sure to update it for you.
  - We appreciate you reaching out to us for assistance with changing the products
    in order number {{Order Number}}. We understand that sometimes adjustments need
    to be made after placing an order. To better assist you, could you please provide
    me with the details of the products you would like to change? This way, we can
    ensure that we process your request accurately and efficiently. Your satisfaction
    is our top priority, and we're here to help you every step of the way.
  - I'm sorry to hear that you're experiencing difficulties with canceling your purchase.
    I understand the importance of canceling purchase {{Order Number}} for you. Please
    rest assured that I'm here to help you find a solution. Could you please provide
    me with more details about the problem you're facing? Once I have a better understanding
    of the issue, I will be able to assist you further and guide you through the cancellation
    process.
- source_sentence: problems with cancelling purchase {{Order Number}}
  sentences:
  - I've grasped that you have a question about cancelling order {{Order Number}}.
    What specific information would you like to know? Are you unsure about the steps
    to cancel the order, or do you have any concerns about the cancellation process?
    Please provide more details so that I can assist you further.
  - Absolutely! I'm here to assist you in checking the termination charges. To do
    that, I'll need some information from you. Could you please provide me with your
    account details or any relevant information so I can access the necessary information
    and provide you with the accurate termination charges?
  - I've been there that you're facing difficulties with canceling your purchase with
    order number {{Order Number}}. Cancelling an order should be a smooth process,
    and I'm here to help you resolve this. To assist you further, I would need some
    specific details about your situation. Could you please provide more information
    about the problems you're encountering while trying to cancel your purchase? This
    will allow me to provide you with the most accurate guidance and ensure a successful
    cancellation. Thank you for reaching out to us, and I appreciate your patience
    as we work towards resolving this issue together.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.32
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.461
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.567
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.674
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.32
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.15366666666666667
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1134
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.0674
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.32
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.461
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.567
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.674
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.4813510411550636
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.42154722222222174
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.43305260585858196
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'problems with cancelling purchase {{Order Number}}',
    "I've been there that you're facing difficulties with canceling your purchase with order number {{Order Number}}. Cancelling an order should be a smooth process, and I'm here to help you resolve this. To assist you further, I would need some specific details about your situation. Could you please provide more information about the problems you're encountering while trying to cancel your purchase? This will allow me to provide you with the most accurate guidance and ensure a successful cancellation. Thank you for reaching out to us, and I appreciate your patience as we work towards resolving this issue together.",
    "Absolutely! I'm here to assist you in checking the termination charges. To do that, I'll need some information from you. Could you please provide me with your account details or any relevant information so I can access the necessary information and provide you with the accurate termination charges?",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7224, 0.0902],
#         [0.7224, 1.0000, 0.0812],
#         [0.0902, 0.0812, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.32       |
| cosine_accuracy@3   | 0.461      |
| cosine_accuracy@5   | 0.567      |
| cosine_accuracy@10  | 0.674      |
| cosine_precision@1  | 0.32       |
| cosine_precision@3  | 0.1537     |
| cosine_precision@5  | 0.1134     |
| cosine_precision@10 | 0.0674     |
| cosine_recall@1     | 0.32       |
| cosine_recall@3     | 0.461      |
| cosine_recall@5     | 0.567      |
| cosine_recall@10    | 0.674      |
| **cosine_ndcg@10**  | **0.4814** |
| cosine_mrr@10       | 0.4215     |
| cosine_map@100      | 0.4331     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 4 tokens</li><li>mean: 13.32 tokens</li><li>max: 24 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 94.12 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:---------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>check bill #85632</code>                                       | <code>No worries at all! I completely understand your uncertainty about how to check the details of your bill with the number #{{Invoice Number}}. Let me guide you through the process step by step. Firstly, are you referring to an online bill or a physical bill received through mail?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>can uhelp me cancelling purchazse {{Order Number}}</code>      | <code>Sure! I can assist you in understanding the termination fees for cancelling your account.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  | <code>I need assistance with canceling order {{Order Number}}</code> | <code>I realized you're looking for assistance with canceling your order with the order number {{Order Number}}. To cancel your order, please follow these steps: 1. Log in to your account on our {{Online Company Portal Info}}. 2. Navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section. 3. Locate the order with the order number {{Order Number}} and click on it. 4. Look for the option labeled '{{Online Order Interaction}}' associated with the purchase. 5. Click on the '{{Online Order Interaction}}' option to initiate the cancellation process. If you encounter any issues or have any questions during the cancellation process, our dedicated support team is available during {{Customer Support Hours}} at {{Customer Support Phone Number}} or through the Live Chat feature on our {{Website URL}}. We are committed to providing you with the assistance you need.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | cosine_ndcg@10 |
|:-----:|:----:|:--------------:|
| 1.0   | 157  | 0.3591         |
| 2.0   | 314  | 0.4626         |
| 3.0   | 471  | 0.4814         |


### Training Time
- **Training**: 1.0 hours
- **Evaluation**: 5.7 minutes
- **Total**: 1.1 hours

### Framework Versions
- Python: 3.12.11
- Sentence Transformers: 5.4.1
- Transformers: 5.8.0
- PyTorch: 2.8.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->