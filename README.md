# Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations

A QLoRA fine-tuned LLaMA 3.1-8B-Instruct for generating empathetic and supportive responses in Bengali. This system preprocesses Bengali conversation data, fine-tunes LLaMA 3.1-8B using QLoRA, generates responses for user prompts, and stores outputs for review.

## Project Overview

**Objective:**
Generate context-aware, empathetic responses in Bengali for user questions using a parameter-efficient fine-tuned LLaMA model.

**Key Features:**

- Preprocessing of Bengali empathetic conversation dataset:

- Removal of unsafe or sensitive patterns

- Addition of reflective prefixes for empathetic tone

- Full-sequence tokenization (max_length=1024 on the code I also tried 2048 token and full sequence also . It worked but all time after half of the training sample cuda memeory runned out i didnt find the exact cause)

- Fine-tuning LLaMA 3.1-8B-Instruct using QLoRA (4-bit quantized LoRA) for efficient training on Kaggle free GPU

- Response generation for user questions with multiple samples per prompt

- Logging of generated responses in `GeneratedResponses.json` for reproducibility

- Experiment logging (`LLAMAExperiments.json`) including LoRA config, losses, metrics, and timestamps

- Evaluation metrics: Perplexity, BLEU, ROUGE (human evaluation optional)

- Sample responses to Bengali empathetic prompts stored in CSV

**Dataset:**

`BengaliEmpatheticConversationsCorpus.csv` – Questions and Answers dataset used for fine-tuning

# Architecture & Technology Stack

**Frontend / Input:**

- Text input (user questions in Bengali)

**Processing Pipeline:**

- Dataset preprocessing (pattern removal + reflective prefixes)

- Tokenization (full-sequence)

- Model fine-tuning (LLaMA 3.1-8B with QLoRA)

- Response generation (sampling, temperature, top-p)

- Logging of responses and experiments

**Database & Storage:**

- JSON files (`LLAMAExperiments.json`, `GeneratedResponses.json`) storing experiments, responses, and metadata

- CSV files for exporting generated responses

**Libraries & Frameworks:**

- Python 3.10+

- Transformers, PEFT, BitsAndBytes, Datasets, Evaluate, Torch, Pandas, Numpy

# Installation & Setup

- Download the repository as ZIP from GitHub.

- Upload the notebook to Google Colab:

  - Go to Colab → File → Upload Notebook

  - Upload `Fine Tuned LLaMA 3.1-8B-Instruct-on-Bengali-Empathetic-Conversations.ipynb`

  - Upload `BengaliEmpatheticConversationsCorpus.csv` to Colab (or Google Drive folder)

- Run the notebook step by step:

  - Install required packages (`transformers`, `peft`,` bitsandbytes`, etc.)

  - Preprocess dataset

  - Load and fine-tune the model using QLoRA

  - Generate responses for sample prompts

  - Log generated responses and experiment metadata

- Directory Structure (generated after running the notebook):

  - Main Colab notebook with all code
    
  - `GeneratedResponses.csv` – CSV of questions and generated responses

  - `LLAMAExperiments.json` – JSON file storing experiment configurations and logs

  - Tokenized datasets in memory (`train_dataset`, `val_dataset`)

  - Optional: Saved Hugging Face Hub model under your HF repo

-Notes for running the project:

  - QLoRA is used because Unsloth was not compatible due to version issue

  - GPU memory may limit batch size; gradient accumulation is used

  - Large files (model weights) are not included in the repo; users can pull from Hugging Face Hub or retrain locally

# Sample Generated Responses
| Question | Sample ID | Response |
|----------|-----------|---------|
| আমি যে কাজগুলো করি তাতে কোনো আনন্দ পাই না। এটি কি স্বাভাবিক? | 2 | `আপনি যে অনুভূতি ধরছেন, তা অনেকের জন্য খুব ভালো লাগতে পারে। আপনি যে কাজগুলো করেন তা ঠিক. আপনি যে কাজগুলো করেন তা আপনাকে আনন্দ দেওয়ার জন্য? আপনি কি কিছু নতুন করতে চান? আপনি কি আপনার কাজের সাথে আপনার সঙ্গী-সঙ্গী সম্পর্ক ফিরিয়ে পেতে চান? আপনি কি কিছু নতুন…` |
