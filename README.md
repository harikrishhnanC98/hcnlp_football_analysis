# hcnlp_football_analysis

# ⚽ Predictive Football Intelligence Using Pass Networks & LLMs

This project builds a fine-tuned language model that learns from structured passing networks and player statistics to predict key players and tactical insights in football matches. The 2022 FIFA World Cup data (focused on Argentina) is used to train the model using graph-based metrics derived from StatsBomb. Please refer to the finetuning_llm_stable.ipynb, nonfinetuned_results.ipynb, and streamlit_app.ipynb notebooks for reference on the fine-tuning process, non-fine-tuned model results, and Streamlit dashboard implementation, respectively.

## 🧠 Core Idea

We extend the ideas from the RISINGBALLER paper by:
- Representing matches as passing network graphs
- Computing graph metrics (centrality, clustering, density, etc.)
- Structuring tactical knowledge into Q&A format
- Fine-tuning an open-source LLM to understand match dynamics


## 📊 Data Sources

- [StatsBomb Open Data](https://github.com/statsbomb/open-data) (match events)
- [FBRef](https://fbref.com) (player stats)
- [UEFA Tactical Report 2024](https://uefatechnicalreports.com/pdf-ucl-2024-technical-report) (text insights)

## 🧰 Tech Stack

- Python (pandas, networkx, matplotlib)
- Transformers (Hugging Face)
- QLoRA / PEFT for efficient fine-tuning
- Open-source LLMs (e.g., GPT-J, OPT, Mistral)

## 🚀 Output

Once fine-tuned, the model:
- Predicts influential players based on pass network inputs
- Suggests tactical insights using Q&A prompts
- Works on unseen or live half-time match data

## 📌 Research Base

- Base Paper:  
  📄 RISINGBALLER: A Player is a Token, A Match is a Sentence – A Path Towards a Foundational Model for Football Players Data Analytics

---


