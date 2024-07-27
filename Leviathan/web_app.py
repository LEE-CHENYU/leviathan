import streamlit as st
import sys 
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx 
import nxviz as nv

from Leviathan.Island_mdp import Island
from Leviathan.Member_mdp import Member
from Leviathan.Analyzer import Analyzer
from time import time
from Leviathan.Land_mdp import Land
from utils import save
import os

rng = np.random.default_rng()
island = Island(20, (5, 5), 2022)
# island = Island.load_from_pickle("data/Nov/15_13-23/180.pkl")

path = save.datetime_dir("../data")
# path = dir+"test_run/"
# os.mkdir(path)
Island._RECORD_PERIOD = 1

st.title("Leviathan Simulation")

st.sidebar.header("Simulation Controls")
num_members = st.sidebar.slider("Number of Members", min_value=1, max_value=100, value=20)
land_shape = st.sidebar.selectbox("Land Shape", options=["(5, 5)", "(10, 10)", "(15, 15)"])
random_seed = st.sidebar.number_input("Random Seed", value=2022)

if st.sidebar.button("Run Simulation"):
    island = Island(num_members, eval(land_shape), random_seed)
    for i in range(10):
        island.new_round(record_path=path)
        island.trade()
        island.land_distribute()
        island.colonize()
        island.consume()
        island.fight()
        island.produce()
        island.reproduce()
        if island.is_dead:
            st.write("The simulation has ended.")
            break

    print(island.print_status())
    st.text(island.print_status(action=True))
    st.pyplot(island.land.plot())
