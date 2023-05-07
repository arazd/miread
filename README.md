# MIReAD
A simple method for learning high-quality representations from scientific documents.

**✨ MIReAD is accepted to ACL 2023 ✨**

This repository contains:
* MIReAD code & evaluation scripts
* link to the pre-trained model weights
* link to the training data
* instructions to use MIReAD

# Loading pretrained model

## 1) via Huggingface Transformers Library

MIReAD weights are available through 🤗 transformers: [https://huggingface.co/arazd/MIReAD](https://huggingface.co/arazd/MIReAD).

Requirement: `pip install --upgrade transformers==4.2`

```python
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('arazd/miread')
model = AutoModel.from_pretrained('arazd/miread')

# concatenate title & abstract
title = 'MIReAD: simple method for learning scientific representations'
abstr = 'Learning semantically meaningful representations from scientific documents can ...'
text = title + tokenizer.sep_token + abstr
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

# classification (getting logits over 2,734 journal classes)
out = model(**inputs)
logits = out.logits

# feature extraction (getting 768-dimensional feature profiles)
# IMPORTANT: use [CLS] token representation as document-level representation (hence, 0th idx)
out = model.bert(**inputs)
representation = out.last_hidden_state[:, 0, :]
```

A sample script to run the model in batch mode on a dataset of papers is provided under `scripts/embed_papers_hf.py`

How to use:
```
CUDA_VISIBLE_DEVICES=0 python scripts/embed_papers_hf.py \
--data-path path/to/paper-metadata.json \
--output path/to/write/output.json \
--batch-size 8
```
