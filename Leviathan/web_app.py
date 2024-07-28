import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
import nxviz as nv
import os
import sys 
from time import time

import streamlit as st

# Ask for API key before proceeding
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
if not api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()
sys.path.append("..")

from Leviathan.Island import Island
from Leviathan.Member import Member
from Leviathan.Land import Land
from Leviathan.Analyzer import Analyzer
from Leviathan.generate_story import generate_story_using_gpt

import openai
openai.api_key = api_key

rng = np.random.default_rng()
path = "./data"
Island._RECORD_PERIOD = 1
Member._DECISION_BACKEND = 'gpt'
Member._PARAMETER_INFLUENCE = 0

st.title("Leviathan Simulation")

st.sidebar.header("Simulation Controls")

rounds = st.sidebar.slider("Number of Rounds", min_value=1, max_value=10, value=3)
num_members = st.sidebar.slider("Number of Members", min_value=1, max_value=10, value=3)
land_shape = st.sidebar.selectbox("Land Shape", options=["(5, 5)", "(10, 10)", "(15, 15)"])
random_seed = st.sidebar.number_input("Random Seed", value=2022)
log_lang = st.sidebar.selectbox("Log Language", options=["English", "中文", "日本語", "Español", "Français", "Deutsch", "Italiano", "Русский"])
action_prob = st.sidebar.slider("Action Probability", min_value=0.0, max_value=1.0, value=0.5)
        
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
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
            
            # Fixed area for plot
            plot_area = st.empty()  # Create a placeholder for the plot
            with plot_area:
                st.pyplot(island.land.plot())
            
            if island.is_dead:
                st.write("The simulation has ended.")
                break
        
    st.session_state.island_finished = True

story_style = st.sidebar.selectbox("Story Style", options=["Homer", "司马迁", "紫式部", "Cervantes", "Shakespeare", "Tolstoy", "Hemingway", "Gabriel García Márquez"])

if "island_finished" not in st.session_state:
    st.session_state.island_finished = False

if st.sidebar.button("Generate Story", disabled=not st.session_state.island_finished):
    with st.spinner("Generating story..."):
            
        story = generate_story_using_gpt(author=story_style, log_path=os.path.join(path, "log.txt"), lang=log_lang)
        st.text(story)

st.sidebar.markdown("### Simulation Description\n"
    """
## Multi-Agent Survival and Social Dynamics Sandbox

This LLM-powered simulation lets you explore emergent behaviors in a virtual island ecosystem. Agents navigate a 2D grid, making decisions based on genetic traits and environmental factors.

### Key Features:

1. **Configurable Parameters:** Adjust rounds, population, island topology, and action probabilities.

2. **Turn-Based Logic:** Agents choose to challenge, offer resources, or reproduce each round.

3. **2D Grid Environment:** Agents claim territory and interact with neighbors.

4. **Social Interaction Model:** Simulates alliance formation, resource sharing, and conflict.

5. **Genetic Algorithm:** Inheritable traits influence agent decision-making.

6. **Event Logging:** Capture all actions and outcomes for data analysis.

7. **Narrative Generation:** Use simulation data to create storylines and scenarios.

### Quick Start:

1. **Initialize Parameters:** Set island size, population, and other variables.
2. **Run Simulation:** Launch and watch agent interactions unfold.
3. **Data Analysis(See our Github project):** Monitor events, alliances, and conflicts in real-time.
4. **Post-Processing:** Generate narratives or analyze trends from simulation data.

> **Note:** This is a simulated environment for research and experimentation. Behaviors do not reflect real-world scenarios but provide a sandbox for studying complex social systems.

Ready to explore emergent behaviors? Launch your simulation and start mining insights!
    """
                    "\n\n#### Author: "
                    "\n\nChenyu Li, Danyang Chen, Mengjun Zhu")