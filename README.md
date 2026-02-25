# Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations

A **QLoRA fine-tuned LLaMA 3.1-8B-Instruct** for generating empathetic and supportive responses in Bengali. This system preprocesses Bengali conversation data, fine-tunes LLaMA 3.1-8B using QLoRA, generates responses for user prompts, and logs outputs for reproducibility.

---

## Project Overview

**Objective:**  
Generate context-aware, empathetic responses in Bengali for user questions using a parameter-efficient fine-tuned LLaMA model.

**Key Features:**
- Preprocessing of Bengali empathetic conversation dataset:
  - Removal of unsafe or sensitive patterns
  - Addition of reflective prefixes to maintain an empathetic tone
  - Full-sequence tokenization (max_length=1024 on the code I also tried 2048 token and full sequence also . It worked but all time after half of the training sample cuda memeory runned out i didnt find the exact cause) 
- Fine-tuning LLaMA 3.1-8B-Instruct using QLoRA (4-bit quantized LoRA) for efficient training on Kaggle free GPU
- Took 4000 sample at first after half sample training like 2000 sample gpu runned out. Then i reduce the sample number but ti sayed the same . after half of the sample training spu runned out
- Response generation for user prompts with multiple samples
- Logging of generated responses in `GeneratedResponses.json` for reproducibility
- Experiment logging (`LLAMAExperiments.json`) including LoRA config, losses, metrics, and timestamps
- Evaluation metrics (planned): Perplexity, BLEU, ROUGE (human evaluation optional)
- Sample responses to Bengali empathetic prompts stored in CSV

**Dataset:**  
`BengaliEmpatheticConversationsCorpus.csv` – Questions and Answers dataset used for fine-tuning

---

## Fine-Tuning Strategies

This project uses **QLoRA (LoRA in 4-bit mode)** to efficiently fine-tune LLaMA 3.1-8B-Instruct on Bengali empathetic conversations. A skeleton for **Unsloth** is included, though it was not executed due to version incompatibilities.

### 1️⃣ QLoRA (Quantized LoRA)

**Overview:**  
QLoRA is a **parameter-efficient fine-tuning method** that allows large language models to be fine-tuned on smaller hardware by **quantizing the model weights to 4-bit** and applying **LoRA adapters** to attention layers. This reduces memory usage while preserving performance.

**Implementation Details:**
- **Base Model:** `meta-llama/Llama-3.1-8b-Instruct`
- **Quantization:** 4-bit NF4 with double quantization (`bnb_4bit_use_double_quant=True`)
- **Target Modules:** Attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Hyperparameters:**  
  - Rank `r=8`  
  - Alpha `16`  
  - Dropout `0.05`
- **Training Optimizations:**  
  - Gradient checkpointing enabled  
  - Mixed precision (FP16) training  
  - Batch size 1, gradient accumulation steps 2

**Benefits:**
- Fits LLaMA 3.1-8B on **Kaggle free GPU**
- Maintains full sequence length (1024 tokens)
- Generates high-quality Bengali empathetic responses
- Efficient experiment logging

**Challenges:**
- GPU memory limitations restricted batch size
- Training sessions may terminate if GPU memory runs out
- Full evaluation metrics could not be generated due to session limits

---

### 2️⃣ Unsloth (Planned but Not Executed)

**Overview:**  
Unsloth is a parameter-efficient fine-tuning method similar to LoRA, intended to provide memory-efficient adaptation for large models.

**Attempted Implementation:**
- Tried installing Unsloth in Kaggle environment
- Version mismatch and dependency issues prevented execution

**Notes:**
- Strategy pattern allows swapping LoRA with Unsloth seamlessly in the codebase
- `UnslothStrategy` class included as a skeleton for future implementation

---

### Summary

- **Used Strategy:** QLoRA  
- **Fallback / Planned Strategy:** Unsloth (not executed)  
- Demonstrates **Strategy Pattern** for flexible fine-tuning approaches

---

## Architecture & Technology Stack

**Frontend / Input:**  
- Text input (user questions in Bengali)

**Processing Pipeline:**  
1. Dataset preprocessing (pattern removal + reflective prefixes)  
2. Tokenization (full-sequence)  
3. Model fine-tuning (LLaMA 3.1-8B with QLoRA)  
4. Response generation (sampling, temperature, top-p)  
5. Logging of responses and experiments

**Database & Storage:**  
- JSON files (`LLAMAExperiments.json`, `GeneratedResponses.json`) storing experiment configurations, responses, and metadata  
- CSV files for exporting generated responses

**Libraries & Frameworks:**  
- Python 3.10+  
- Transformers, PEFT, BitsAndBytes, Datasets, Evaluate, Torch, Pandas, Numpy

---

## Installation & Setup

1. **Download the repository** as ZIP from GitHub.

2. **Upload notebook to Google Colab / Kaggle:**  
   - Upload `Fine Tuned LLaMA 3.1-8B-Instruct-on-Bengali-Empathetic-Conversations.ipynb`  
   - Upload `BengaliEmpatheticConversationsCorpus.csv`  
   - For Hugging Face models, provide a valid HF access token

3. **Run the notebook step by step:**  
   - Install required packages (`transformers`, `peft`, `bitsandbytes`, etc.)  
   - Preprocess dataset  
   - Load and fine-tune model using QLoRA  
   - Generate responses for sample prompts  
   - Log responses and experiment metadata

4. **Directory Structure (generated after running notebook):**  
   - Main notebook  
   - `GeneratedResponses.csv` – CSV of questions and generated responses  
   - `LLAMAExperiments.json` – JSON logs of experiments  
   - Tokenized datasets (`train_dataset`, `val_dataset`) in memory  
   - Optional: Saved Hugging Face Hub model

**Notes:**  
- QLoRA is used because Unsloth was not compatible due to version issues  
- GPU memory may limit batch size; gradient accumulation is used  
- Large files (model weights) are not included in the repo; users can pull from HF Hub or retrain locally

---

## Sample Generated Responses

| Question | Sample ID | Response |
|----------|-----------|---------|
| আমি যে কাজগুলো করি তাতে কোনো আনন্দ পাই না। এটি কি স্বাভাবিক? | 2 | `আপনি যে অনুভূতি ধরছেন, তা অনেকের জন্য খুব ভালো লাগতে পারে। আপনি যে কাজগুলো করেন তা ঠিক. আপনি কি কিছু নতুন করতে চান? আপনি কি আপনার কাজের সাথে আপনার সঙ্গী-সঙ্গী সম্পর্ক ফিরিয়ে পেতে চান? আপনি কি কিছু নতুন…` |
