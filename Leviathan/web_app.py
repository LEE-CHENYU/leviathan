import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
import nxviz as nv
import os
import sys 
from time import time

import streamlit as st
sys.path.append("..")

from Leviathan.Island import Island
from Leviathan.Member import Member
from Leviathan.Land import Land
from Leviathan.Analyzer import Analyzer
from Leviathan.generate_story import generate_story_using_gpt

rng = np.random.default_rng()
path = "./data"
Island._RECORD_PERIOD = 1
Member._DECISION_BACKEND = 'gpt'
Member._PARAMETER_INFLUENCE = 0

st.title("Leviathan Simulation")

st.sidebar.header("Simulation Controls")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")

with open("./Leviathan/api_key.py", "w") as f:
    f.write("import openai\n")
    f.write(f"{api_key}\n")

rounds = st.sidebar.slider("Number of Rounds", min_value=1, max_value=10, value=3)
num_members = st.sidebar.slider("Number of Members", min_value=1, max_value=10, value=3)
land_shape = st.sidebar.selectbox("Land Shape", options=["(5, 5)", "(10, 10)", "(15, 15)"])
random_seed = st.sidebar.number_input("Random Seed", value=2022)
log_lang = st.sidebar.selectbox("Log Language", options=["English", "中文", "日本語", "Español"])
action_prob = st.sidebar.slider("Action Probability", min_value=0.0, max_value=1.0, value=0.5)
        
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        os.makedirs(path, exist_ok=True)
        island = Island(num_members, eval(land_shape), path, random_seed, log_lang)

        for i in range(rounds):
            island.new_round()
            island.get_neighbors()
            island.trade(action_prob)
            island.land_distribute(action_prob)
            island.colonize()
            island.consume()
            island.fight(action_prob)
            island.produce()
            island.reproduce(action_prob)
            island.record_statistics()
            island.log_status(
                action=True,
                summary=True,
                members=True,
                log_instead_of_print=True,
            )
            
            st.text(island.log_status(action=True))
            log_file_path = os.path.join(path, "log.txt")
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as log_file:
                    log_contents = log_file.read()
                    st.text(log_contents)
            else:
                st.text("Log file not found.")
            
            # Fixed area for plot
            plot_area = st.empty()  # Create a placeholder for the plot
            with plot_area:
                st.pyplot(island.land.plot())
            
            if island.is_dead:
                st.write("The simulation has ended.")
                break
        
    st.session_state.island_finished = True

story_style = st.sidebar.selectbox("Story Style", options=["Homer", "司马迁", "紫式部", "Cervantes"])

if "island_finished" not in st.session_state:
    st.session_state.island_finished = False

if st.sidebar.button("Generate Story", disabled=not st.session_state.island_finished):
    with st.spinner("Generating story..."):
        story = generate_story_using_gpt(author=story_style, log_path=os.path.join(path, "log.txt"), lang=log_lang)
        st.text(story)

st.sidebar.markdown("### Simulation Description\n"
                    "This social simulation models the interactions and behaviors of members that powered by LLM within an island ecosystem. "
                    "You can control various parameters such as the number of rounds, members, land shape, and action probabilities. "
                    "The simulation will log the actions and outcomes, allowing you to analyze the dynamics of the ecosystem. "
                    "Try to generate a story based on the simulation results to explore the narrative possibilities!"
                    "\n\n#### Author: "
                    "\n\nChenyu Li, Danyang Chen, Mengjun Zhu")