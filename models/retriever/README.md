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
- source_sentence: help me cancelling purchase {{Order Number}}
  sentences:
  - 'I comprehend your confusion about finding the bill from {{Salutation}} {{Client
    Last Name}}. Let''s get you the information you need! To view the bill from {{Person
    Name}}, you can try the following steps: 1. Log in to your account on our website
    or mobile app. 2. Navigate to the "Billing" section or a similar tab. 3. Look
    for an option like "View Invoices" or "Billing History." 4. Filter the invoices
    by date or search for {{Person Name}} specifically. 5. Once you locate the bill,
    you can click on it to view the details. If you encounter any difficulties or
    need further assistance, please don''t hesitate to let me know. I''m here to help
    you every step of the way!'
  - I've come to understand that you have a question regarding cancelling purchase
    {{Order Number}}. I'm here to provide you with the information you need. Please
    go ahead and let me know what specific question you have about cancelling your
    purchase. I'll be happy to assist you further.
  - To view the early exit charges, you can access your account online and navigate
    to the "Account Details" section. From there, you should be able to find all the
    information regarding the early exit charges associated with your plan.
- source_sentence: i lost the bills from {{Person Name}} can ya ifnd it for me
  sentences:
  - Thank you for contacting to us regarding the request to switch several items from
    order number {{Order Number}}. I completely understand that you may have changed
    your mind or found other products that better suit your needs. To assist you further,
    could you please provide me with the specific items you would like to switch and
    the alternative items you would like to replace them with? This will allow us
    to process your request efficiently and ensure your satisfaction.
  - 'Sure! I completely understand your desire to quickly access your invoice with
    the number #{{Invoice Number}}. It can be frustrating when you need to find important
    information swiftly. To give you a quick look at your invoice #{{Invoice Number}},
    you can try accessing your account portal on our website. From there, navigate
    to the "Invoices" or "Billing" section, where you should find a list of all your
    invoices, including the one with the number #{{Invoice Number}}. If you encounter
    any issues or need further assistance, please don''t hesitate to let me know.
    Your satisfaction is my top priority!'
  - I grasp that you have lost the invoices from {{Person Name}} and you need assistance
    in locating them. I apologize for the inconvenience caused. To ensure a prompt
    resolution, could you please provide me with some additional information such
    as the date range or any specific details about the invoices? This will help me
    narrow down the search and assist you better. Rest assured, I'm here to help you
    find those lost bills and provide you with the necessary support.
- source_sentence: i want supports trying to update my shippingaddress
  sentences:
  - 'I catch on to the fact that you are looking to cancel your purchase with the
    order number {{Order Number}}. I apologize for any inconvenience this may have
    caused. To cancel your order, please follow these steps: 1. Log into your {{Online
    Company Portal Info}} using your credentials. 2. Navigate to the ''{{Online Order
    Interaction}}'' or ''{{Online Order Interaction}}'' section. 3. Locate the purchase
    with the order number {{Order Number}} and click on it to view the details. 4.
    Look for the option labeled ''{{Online Order Interaction}}'' associated with your
    purchase and select it. 5. Complete any additional steps or provide feedback if
    prompted by the system. If you encounter any difficulties or have further questions,
    our dedicated team is available during {{Customer Support Hours}}. You can reach
    us at {{Customer Support Phone Number}} or through the Live Chat feature on our
    {{Website URL}}. Your satisfaction is our top priority, and we are here to assist
    you through the cancellation process.'
  - I apologize for the inconvenience you're facing while trying to correct your shipping
    address. Rest assured, I'll do my best to assist you with resolving this issue.
    To get started, could you please provide me with your current shipping address
    and the correct address you'd like to update it to?
  - I'll do my best! I'm here to assist you in updating your delivery address. Please
    provide me with your current address and the new address you want to update it
    to, and I'll take care of it for you.
- source_sentence: how can i see the withdrawal fees
  sentences:
  - We understand that you would like to edit purchase number {{Order Number}}. Editing
    a purchase is possible in certain cases, depending on the status of the order
    and the specific changes you would like to make. To provide you with accurate
    information, could you please specify the specific edits you have in mind? This
    will help us determine the feasibility and guide you accordingly. We are committed
    to ensuring your satisfaction and will do our best to assist you with your request.
  - Thank you for reaching out! I understand your uncertainty about how to initiate
    a conversation with an assistant. Allow me to provide you with the steps you can
    follow to connect with a human representative. First, please check if there is
    a "Chat" or "Contact" option available on our website or app. You can usually
    find this option at the bottom of the page or in the menu. If you cannot locate
    it, please try searching for "Support" or "Help" on our website or app, as these
    sections often provide ways to communicate with a human representative. If you
    still cannot find a direct contact method, please provide me with the name of
    the product or service you need assistance with, and I will guide you to the appropriate
    support channel. Remember, I am here to help you every step of the way.
  - I'm on it! I can provide you with information about the withdrawal fees. When
    it comes to withdrawals from your account, there may be certain fees associated
    with the transaction. These fees can vary depending on factors such as the type
    of account you have and the withdrawal method you choose. To get specific details
    about the withdrawal fees applicable to your account, I recommend checking the
    "Fees and Charges" section in your online profile or contacting our customer service
    team directly. They will be able to provide you with accurate and up-to-date information
    regarding the withdrawal fees for your particular account.
- source_sentence: where do I update the delivery address?
  sentences:
  - Thank you for bringing this to our attention. I understand that you would like
    to swap some products in your order with the number {{Order Number}}. Our team
    is here to assist you with this request. To ensure we handle the swap accurately,
    could you please provide the specific details of the products you would like to
    exchange and the replacements you would like to receive? This will enable us to
    quickly process the swap and ensure your satisfaction.
  - 'I''ll get right on it! I''m here to assist you in checking the invoice from your
    purchase made 4 purchases ago. To provide you with accurate information, could
    you please provide me with the following details: 1. Your full name. 2. The approximate
    date of the purchase. 3. Any additional information that can help me locate the
    invoice, such as the product or service purchased. Once I have these details,
    I''ll do my best to help you locate the invoice from your purchase 4 purchases
    ago.'
  - I apologize for the inconvenience. Can you please provide me with more details
    about the issue you are facing while trying to update your address? This will
    help me assist you better in resolving the problem.
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
      value: 0.331
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.503
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.575
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.69
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.331
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.16766666666666666
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.115
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.069
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.331
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.503
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.575
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.69
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.49754370475418064
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.437567063492063
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.44769279874739837
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
    'where do I update the delivery address?',
    'I apologize for the inconvenience. Can you please provide me with more details about the issue you are facing while trying to update your address? This will help me assist you better in resolving the problem.',
    "I'll get right on it! I'm here to assist you in checking the invoice from your purchase made 4 purchases ago. To provide you with accurate information, could you please provide me with the following details: 1. Your full name. 2. The approximate date of the purchase. 3. Any additional information that can help me locate the invoice, such as the product or service purchased. Once I have these details, I'll do my best to help you locate the invoice from your purchase 4 purchases ago.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.3820, 0.0726],
#         [0.3820, 1.0000, 0.0135],
#         [0.0726, 0.0135, 1.0000]])
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
| cosine_accuracy@1   | 0.331      |
| cosine_accuracy@3   | 0.503      |
| cosine_accuracy@5   | 0.575      |
| cosine_accuracy@10  | 0.69       |
| cosine_precision@1  | 0.331      |
| cosine_precision@3  | 0.1677     |
| cosine_precision@5  | 0.115      |
| cosine_precision@10 | 0.069      |
| cosine_recall@1     | 0.331      |
| cosine_recall@3     | 0.503      |
| cosine_recall@5     | 0.575      |
| cosine_recall@10    | 0.69       |
| **cosine_ndcg@10**  | **0.4975** |
| cosine_mrr@10       | 0.4376     |
| cosine_map@100      | 0.4477     |

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
  | details | <ul><li>min: 4 tokens</li><li>mean: 13.55 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 94.98 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>how to switch a product of order {{Order Number}}?</code>         | <code>Thank you for contacting to us about your request to switch a product in purchase number {{Order Number}}. We understand the importance of ensuring your order meets your expectations. To assist you with this, could you please provide us with the details of the product you would like to switch and the new product you would like to replace it with? This will enable us to review your request and provide you with the necessary guidance. We appreciate your patience and look forward to resolving this matter for you.</code>                                                              |
  | <code>I would like to switch something of order {{Order Number}}</code> | <code>We've received your message to us with your request. I understand that you would like to switch something in order number {{Order Number}}. To better assist you, could you please provide more details about the item you would like to switch and the item you would like to switch it with? This way, we can ensure we find the best solution for you. Your satisfaction is our top priority, and we're here to help make any necessary adjustments to your order.</code>                                                                                                                            |
  | <code>I can't edit order {{Order Number}}</code>                        | <code>Oh, what a thrilling adventure we are about to embark on! The task of editing your esteemed account is one that I am more than thrilled to assist you with. Please accept my sincere apologies, but I'm afraid I can't directly edit your account details. However, I can guide you through the process, ensuring that you have all the necessary information to edit your account successfully. May I kindly ask which specific details you would like to update? Let's dive into this empowering journey together, changing your account information in a way that perfectly suits your needs.</code> |
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
| 1.0   | 157  | 0.3425         |
| 2.0   | 314  | 0.4656         |
| 3.0   | 471  | 0.4975         |


### Training Time
- **Training**: 1.4 minutes
- **Evaluation**: 9.2 seconds
- **Total**: 1.5 minutes

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