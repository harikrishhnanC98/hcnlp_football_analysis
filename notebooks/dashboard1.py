import streamlit as st
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import json
import threading
import os
from pyngrok import ngrok
import time

# --- New Preset for Score Prediction Prompt ---
score_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.

**Assume the second half of the final has not happened yet.**
DO NOT:
- Invent or guess numbers (e.g., "75% possession" or "passes made")
- Mention the goalkeeper (exclude Emiliano Martínez)
- Refer to a player without citing their stats
- Skip any section or provide vague reasoning

Below given are the players and their positions where FW-forward, LM-Left midfield, CM-Central midfield, RM-Right midfield, LB-Left back, CB-Centre back, RB-Right back, GK-Goalkeeper:
- **Do not mix up player positions. Always follow the positions mentioned below**
Lautaro Martínez  FW
Lionel Messi  FW - Starting 11
Papu Gómez    LM
Julián Álvarez  LM - Starting 11
Leandro Paredes CM
Enzo Fernández  CM - Starting 11
Rodrigo De Paul CM - Starting 11
Ángel Di María  RM - Starting 11
Nicolás Tagliafico  LB - Starting 11
Marcos Acuña  LB
Nicolás Otamendi  CB - Starting 11
Cristian Romero   CB - Starting 11
Lisandro Martínez CB
Nahuel Molina RB - Starting 11
Emiliano Martínez GK - Starting 11

**The player names with '- Starting 11' indicate the players who started the match.**

**First Half Formation**
Argentina Formation - 4-3-3

**First Half Goals**
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.

**Key First-Half Statistics**
Possession: Argentina held 55% of the ball, while France had 45%.
Shots on Target: Argentina registered 5 shots on goal; France had none.
Total Shots: Argentina attempted 10 shots; France managed 7.
Corners: Argentina earned 6 corners; France had 5.
Fouls Committed: Argentina committed 26 fouls; France committed 19.
Yellow Cards: Argentina received 5 yellow cards; France received 3.

Please respond using the following structure and only based on the input data provided:

**1. Influential Passes and Players**
- Identify specific passes leading to goals or scoring opportunities.
- Include player names, positions, and relevant pass metrics (passes made, received, centrality values).

**2. Tactical Suggestions for the Second Half**
- Recommend:
  a) Substitutions (who and why, based on passes made/received and centrality).
  b) Pass totals (from graph metrics)
  c) Positional changes (e.g., shift Di María inward, or allow Fernández more forward freedom).
  d) Formation tweaks (e.g., overload left side if pass clustering is higher there).
- Always support suggestions with exact stats.

**3. Positive/Favorable Situations**
- Highlight passing patterns that led to shots, space creation, or domination in zones.

**4. Overall Inference**
- Assess Argentina’s first-half using:
  a) Possession (%)
  b) Network density and clustering coefficient
  c) Player-specific centrality metrics
- Provide:
  a) 2 specific suggestions to improve second-half passing efficiency
  b) Predict which player (with stats) is most likely to impact the second half positively.

### Final Instructions
- **Be careful and provide only the actual values from the input data**
- **Use accurate metrics when discussing player impact**
- **Avoid general football commentary**
- **Exclude Damián Emiliano Martínez (GK) from tactical analysis**
- **Strictly follow the input and do not hallucinate new data**
- **Each player mention must include at least two metrics: passes made, passes received, degree, betweenness, closeness, pagerank, eigenvector, or clustering.**
- **Omit general statements like "played well" or "strong control".**
###Response
""".strip()

possession_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.
**Assume the second half of the final has not happened yet.**

Below given are the players and their positions where FW-forward, LM-Left midfield, CM-Central midfield, RM-Right midfield, LB-Left back, CB-Centre back, RB-Right back, GK-Goalkeeper:
Lionel Messi  FW - Starting 11
Julián Álvarez  LM - Starting 11
Enzo Fernández  CM - Starting 11
Rodrigo De Paul CM - Starting 11
Ángel Di María  RM - Starting 11
Nicolás Tagliafico  LB - Starting 11
Nicolás Otamendi  CB - Starting 11
Cristian Romero   CB - Starting 11
Nahuel Molina RB - Starting 11

**First Half Formation**
Argentina Formation in the first-half- 4-3-3
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.
Possession: Argentina held 55% of the ball, while France had 45%.
Follow these Instructions:
The player names with '- Starting 11' indicate the players who started the match.
There is no information about France stats, so don't use them.
Strictly follow the input and do not hallucinate new data
Using the First-Half Pass Network Summary and First-Half Graph-Level Summary from the 2022 World Cup final (Argentina vs. France), provide the following details for Argentina:
  1. Predict scoreline for second-half based on historical data and first-half network and graph-level summary.
  2. Predict top player based on pass network summary and graph-level summary
  3. Predict expected possesion metric for the second-half
  4. Predict most favourable passing lane and the players included in the passing lane for the second-half
  5. 1 tip to improve top playmaker's contribution
  6. Predict whether Argentina should stick to the same formation 4-3-3 or change formation to suit better passing to properly utilize underutilized players.

###Response
""".strip()

top_player_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.
**First Half Formation**
Argentina Formation in the first-half- 4-3-3
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.
Possession: Argentina held 55% of the ball, while France had 45%.

**Below given are the players who are Argentina's starting 11:**
Lionel Messi  FW - Starting 11
Julián Álvarez  FW - Starting 11
Enzo Fernández  CM - Starting 11
Rodrigo De Paul CM - Starting 11
Ángel Di María  FW - Starting 11
Nicolás Tagliafico  LB - Starting 11
Nicolás Otamendi  CB - Starting 11
Cristian Romero   CB - Starting 11
Nahuel Molina RB - Starting 11
**These are the players who are on Argentina's bench**:
1. Lautaro Martínez (FW):
   - Passes made: 12.8, Passes received: 10.2
   - Degree: 0.59, Betweenness: 0.31, Closeness: 0.56
   - PageRank: 0.47, Eigenvector: 0.50, Clustering: 0.23
2. Lisandro Martínez (CB) - Bench player:
   - Passes made: 15.4, Passes received: 13.7
   - Degree: 0.64, Betweenness: 0.27, Closeness: 0.60
   - PageRank: 0.45, Eigenvector: 0.52, Clustering: 0.28
3. Leandro Paredes (CM)- Bench player:
   - Passes made: 28.5, Passes received: 24.1
   - Degree: 0.78, Betweenness: 0.48, Closeness: 0.75
   - PageRank: 0.66, Eigenvector: 0.68, Clustering: 0.34
4. Marcos Acuña (LB)- Bench player:
   - Passes made: 23.7, Passes received: 21.2
   - Degree: 0.71, Betweenness: 0.38, Closeness: 0.67
   - PageRank: 0.58, Eigenvector: 0.60, Clustering: 0.30
5.  Gonzalo Montiel(RB)- Bench player:
   - Degree Centrality: 0.823, Betweenness Centrality: 0.00875, Closeness Centrality: 0.632
   - Eigenvector Centrality: 0.1766, PageRank: 0.03925, Clustering: 0.9222
   - Passes Made: 22.33, Passes Received: 21.67
6. Germán Pezzella(CB)- Bench player:
    - Degree Centrality: 0.9646, Betweenness Centrality: 0.01539, Closeness Centrality: 0.6739
    - Eigenvector Centrality: 0.1979, PageRank: 0.0333, Clustering: 0.8984
    - Passes Made: 24.0, Passes Received: 20.5

**As Argentina is leading 2-0, focus should be on reducing the risk of counter-attacks and maintaining control of possession.**

**Please respond using the following structure and by comparing First-Half metrics of starting players provided with the average metrics of bench players:**
1. Should Nicolás Otamendi(CB) be replaced by Germán Pezzella(CB) or Lisandro Martínez (CB)?
2. Should Nahuel Molina(RB) be replaced by Gonzalo Montiel(RB)?
3. Should Nicolás Tagliafico(LB) be replaced by Marcos Acuña (LB)?
4. Should Julián Álvarez(FW) be replaced by Lautaro Martínez (FW)?
6. How many of these substitutions should Argentina make?
7. Should Argentina's 4-3-3 formation be changed in the second half to accomodate these substitutions?

###Response
""".strip()

# New prompts for the small cards
expected_scoreline_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.

**Assume the second half of the final has not happened yet.**

**First Half Formation**
Argentina Formation - 4-3-3

**First Half Goals**
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.

**Key First-Half Statistics**
Possession: Argentina held 55% of the ball, while France had 45%.
Shots on Target: Argentina registered 5 shots on goal; France had none.
Total Shots: Argentina attempted 10 shots; France managed 7.
Corners: Argentina earned 6 corners; France had 5.
Fouls Committed: Argentina committed 26 fouls; France committed 19.
Yellow Cards: Argentina received 5 yellow cards; France received 3.

Predict the expected scoreline for the second half in the format: Arg (score) - Fra (score).

###Response
""".strip()

expected_possession_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.

**Assume the second half of the final has not happened yet.**

**First Half Formation**
Argentina Formation - 4-3-3

**First Half Goals**
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.

**Key First-Half Statistics**
Possession: Argentina held 55% of the ball, while France had 45%.
Shots on Target: Argentina registered 5 shots on goal; France had none.
Total Shots: Argentina attempted 10 shots; France managed 7.
Corners: Argentina earned 6 corners; France had 5.
Fouls Committed: Argentina committed 26 fouls; France committed 19.
Yellow Cards: Argentina received 5 yellow cards; France received 3.

Predict the expected possession for Argentina in the second half in the format: Arg (possession value expected).

###Response
""".strip()

expected_top_player_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.

**Assume the second half of the final has not happened yet.**

**First Half Formation**
Argentina Formation - 4-3-3

**First Half Goals**
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.

**Key First-Half Statistics**
Possession: Argentina held 55% of the ball, while France had 45%.
Shots on Target: Argentina registered 5 shots on goal; France had none.
Total Shots: Argentina attempted 10 shots; France managed 7.
Corners: Argentina earned 6 corners; France had 5.
Fouls Committed: Argentina committed 26 fouls; France committed 19.
Yellow Cards: Argentina received 5 yellow cards; France received 3.

Predict the expected top player of Argentina in the second half.

###Response
""".strip()

expected_top_substitution_prompt = """
You are an expert in football data analysis, specialized in predicting match outcomes based on pass network data. You have been fine-tuned on pass networks (created using StatsBomb player pass stats), match statistics, and text reports from Argentina's previous six matches in the 2022 World Cup.

**Assume the second half of the final has not happened yet.**

**First Half Formation**
Argentina Formation - 4-3-3

**First Half Goals**
23rd minute: Lionel Messi converted a penalty after Ángel Di María was fouled in the box by Ousmane Dembélé.
36th minute: Ángel Di María finished a swift team move, doubling Argentina's lead. Key pass Assist was made by Alexis Mac Allister.

**Key First-Half Statistics**
Possession: Argentina held 55% of the ball, while France had 45%.
Shots on Target: Argentina registered 5 shots on goal; France had none.
Total Shots: Argentina attempted 10 shots; France managed 7.
Corners: Argentina earned 6 corners; France had 5.
Fouls Committed: Argentina committed 26 fouls; France committed 19.
Yellow Cards: Argentina received 5 yellow cards; France received 3.

Predict the expected top substitution of Argentina in the second half.

###Response
""".strip()

# Set the page config first, before any other Streamlit commands
st.set_page_config(page_title="World Cup Match Analyzer", layout="wide")

@st.cache_resource
def load_model():
    adapter_path = "/content/hcnlp_football_analysis/finetuned_llama3.2_augmented"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    model.eval()
    torch.cuda.empty_cache()
    return model, tokenizer

model, tokenizer = load_model()

with open("/content/hcnlp_football_analysis/test_data/match_7_first_half.json", "r") as f:
    data = json.load(f)

pass_network_text = ""
for player, stats in data["pass_network"].items():
    pass_network_text += (
        f"- {player}: "
        f"{stats['passes_made']} passes made, "
        f"{stats['passes_received']} received, "
        f"Degree: {stats['degree_centrality']:.2f}, "
        f"Betweenness: {stats['betweenness_centrality']:.2f}, "
        f"Closeness: {stats['closeness_centrality']:.2f}, "
        f"PageRank: {stats['pagerank']:.2f}, "
        f"Eigenvector: {stats['eigenvector_centrality']:.2f}, "
        f"Clustering: {stats['clustering']:.2f}"
    )

graph = data["graph_metrics"]
graph_text = (
    f"The overall passing network had {graph['num_nodes']} players and "
    f"{graph['num_edges']} total passes (edges). "
    f"Network density: {graph['density']:.3f}, "
    f"Average clustering coefficient: {graph['avg_clustering']:.3f}."
)

# Combined base prompt
base_prompt = f"Match Context:\n{graph_text}\n\nPlayer Stats:\n{pass_network_text}\n\n"

def call_model(user_prompt, max_tokens=256):
    full_prompt = base_prompt + user_prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
      output = model.generate(
          **inputs,
          max_new_tokens=512,
          do_sample=True,
          temperature=0.1,
          top_p=0.9,
          repetition_penalty=1.1,
          eos_token_id=tokenizer.eos_token_id,
      )
    # Decode output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(full_prompt, "").strip()

st.title("🏆 World Cup 2022: Match Insight Dashboard1")
st.subheader("Argentina vs France — Final (First Half Analysis)")

st.markdown("### 🔧 Prompt Console (Editable Templates Only)")
with st.expander("View/Edit Prompts Used in Cards"):
    score_prompt = st.text_area("Key Player Role Analysis", value=score_prompt, height=80)
    possession_prompt = st.text_area("Tactical Forecast", value=possession_prompt, height=80)
    top_player_prompt = st.text_area("Substitution Impact", value=top_player_prompt, height=80)

# New section for small cards
st.markdown("### 📊 Quick Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 🎯 Expected Scoreline")
    st.success(call_model(expected_scoreline_prompt))

with col2:
    st.markdown("#### ⚽ Expected Possession")
    st.info(call_model(expected_possession_prompt))

with col3:
    st.markdown("#### 🏅 Top Player")
    st.warning(call_model(expected_top_player_prompt))

with col4:
    st.markdown("#### 🔄 Top Substitution")
    st.error(call_model(expected_top_substitution_prompt))

st.markdown("### 📊 AI-Based Predictions")
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("#### 🔮 🎯 Key Player Role Analysis & Second-Half Influence Prediction")
    st.success(call_model(score_prompt))

with col6:
    st.markdown("#### ⚽ Tactical Forecast: Scoreline, Possession Trends & Playmaker Strategy")
    st.info(call_model(possession_prompt))

with col7:
    st.markdown("#### 🔄 Substitution Impact & Game Management Strategy")
    st.warning(call_model(top_player_prompt))

st.markdown("### 🧠 First-Half Pass Network Visualization")
st.image(
    "/content/hcnlp_football_analysis/test_data/7_Arg_vs_Fra_Final_first_half_network.png",
    caption="(Insert custom pass network graphic here)",
    use_column_width=True
)

st.markdown("### 📄 Generated Match Report")
report_prompt = "Generate a detailed report of Argentina vs France using the first-half data only."
report = call_model(report_prompt, max_tokens=512)
st.text_area("Model Report Output", value=report, height=200)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

# Launch Streamlit app
def run_app():
    os.system('streamlit run /content/hcnlp_football_analysis/notebooks/dashboard.py')

thread = threading.Thread(target=run_app)
thread.start()

# Wait a few seconds for Streamlit to boot
time.sleep(5)

# Setup ngrok tunnel
public_url = ngrok.connect(addr=8501, proto="http")
print(f"Public URL: {public_url}")
