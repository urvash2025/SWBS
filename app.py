import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import json

# Streamlit UI Setup
st.title("Smart WBS Maker")
st.write("Generate a structured Work Breakdown Structure (WBS) using AI")

# User Inputs
activity_name = st.text_input("Enter Activity Name:")
project_type = st.selectbox("Select Project Type", ["Infrastructure", "High-Rise", "Road", "Other"])
project_phase = st.selectbox("Select Project Phase", ["Design", "Procurement", "Construction", "Testing"])

# OpenAI API Key (Set manually or from secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize GPT-4 model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["activity_name", "project_type", "project_phase"],
    template="""
    You are an expert in project management. Generate a Work Breakdown Structure (WBS) with proper numbering and hierarchy.
    
    **Activity Name:** {activity_name}  
    **Project Type:** {project_type}  
    **Project Phase:** {project_phase}  
    
    **Format the WBS as follows:**  
    - Phases (color-coded representation)  
    - Tasks & Subtasks (structured hierarchy)  
    - Proper numbering (1.1, 1.2, etc.)
    
    Output should be structured clearly for both text and visual representation.
    """
)

# Create Langchain LLM Chain
wbs_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate WBS
def generate_wbs(activity_name, project_type, project_phase):
    response = wbs_chain.run(activity_name=activity_name, project_type=project_type, project_phase=project_phase)
    return response

# Generate WBS Button
if st.button("Generate WBS"):
    if activity_name:
        wbs_output = generate_wbs(activity_name, project_type, project_phase)
        st.subheader("Generated WBS:")
        st.text(wbs_output)

        # Convert to JSON
        wbs_json = json.dumps({"WBS": wbs_output}, indent=4)
        st.download_button("Download WBS as JSON", data=wbs_json, file_name="WBS_Structure.json", mime="application/json")
        
        # Convert to Excel
        wbs_lines = [line.strip().split(" ", 1) for line in wbs_output.split("\n") if line.strip()]
        df = pd.DataFrame(wbs_lines, columns=["Task No.", "Description"])
        excel_path = "WBS_Structure.xlsx"
        df.to_excel(excel_path, index=False)
        st.download_button("Download WBS as Excel", data=open(excel_path, "rb"), file_name="WBS_Structure.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # Create a hierarchical WBS visualization
        G = nx.DiGraph()
        prev_node = None
        for line in wbs_output.split("\n"):
            if line.strip():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    G.add_edge(parts[0], parts[1])
                    prev_node = parts[1]
        
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10, font_weight="bold")
        plt.title("Hierarchical Work Breakdown Structure (WBS)")
        st.pyplot(plt)
    else:
        st.warning("Please enter an activity name.")
