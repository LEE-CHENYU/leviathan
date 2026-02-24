# import sys 
# sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx 
import nxviz as nv

from Leviathan.Island import Island
from Leviathan.Member import Member
from Leviathan.Analyzer import Analyzer, Tracer
from time import time
from Leviathan.Land import Land
from utils import save
import os

import glob
import pickle
island_stamp = '23_16-53'
pickle_folder = f'/Users/chenyusu/vscode/leviathan/data/Mar/{island_stamp}'

tracer = Tracer.load_from_pickle_folder(pickle_folder, 500, 550)

tracer.land_animation()