# ⚽ Predictive Football Intelligence Using Pass Networks & LLMs

This project builds a fine-tuned language model that learns from structured passing networks and player statistics to predict key players and tactical insights in football matches. The 2022 FIFA World Cup data (focused on Argentina) is used to train the model using graph-based metrics derived from StatsBomb. Please refer to the `finetuning_llm_stable.ipynb`, `nonfinetuned_results.ipynb`, and `streamlit_app.ipynb` notebooks in the root directory for details on the fine-tuning process, non-fine-tuned model results, and Streamlit dashboard implementation, respectively.

## 🧠 Core Idea

We extend the ideas from the RISINGBALLER paper by:

- Representing matches as passing network graphs
- Computing graph metrics (centrality, clustering, density, etc.)
- Structuring tactical knowledge into Q&A format
- Fine-tuning an open-source LLM to understand match dynamics

## 📊 Data Sources

- StatsBomb Open Data (2022 FIFA World Cup match events)
- FBRef (2022 player stats)
- UEFA Tactical Report 2024 (supplementary tactical insights)

## 🧰 Tech Stack

- Python (pandas, networkx, matplotlib)
- Transformers (Hugging Face)
- QLoRA / PEFT for efficient fine-tuning
- Open-source LLMs (Llama3.2 1b Instruct)

## 🚀 Output

Once fine-tuned, the model:

- Predicts influential players based on pass network inputs
- Suggests tactical insights using Q&A prompts
- Works on unseen or live half-time match data

## 📌 Research Base

- **Base Paper**:
  📄 RISINGBALLER: A Player is a Token, A Match is a Sentence – A Path Towards a Foundational Model for Football Players Data Analytics

## How to Use This GitHub

### Key Folders:

- `data`: Contains all 2022 FIFA World Cup data (pass networks, match stats, text reports, augmented data for fine-tuning).
- `outputs`: Holds predictions, processed 2022 match data, and Q&A outputs.

### Main Notebooks:

- Run `finetuning_llm_stable.ipynb` to fine-tune the model and see all the prediction results.
- Run `nonfinetuned_results.ipynb` to see non-fine-tuned model outputs and see all the prediction results. Additionally contains the comparison graph of both results.
- Run `streamlit_app.ipynb` for the interactive dashboard UI.

### Dependencies:

- Install Python packages: `pip install torch transformers datasets pandas networkx matplotlib streamlit ngrok`.
- Ensure Hugging Face Transformers and QLoRA are set up for LLM fine-tuning.

### Steps:

- Clone the repo: `[git clone https://github.com/harikrishhnanC98/hcnlp_football_analysis.git](https://github.com/harikrishhnanC98/hcnlp_football_analysis.git)`.
- Install dependencies in a virtual environment.
- Execute notebooks in order (finetuning_llm_stable.ipynb, streamlit_app.ipynb, nonfinetuned_results.ipynb) using Jupyter or Colab.
