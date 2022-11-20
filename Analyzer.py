import numpy as np 
import networkx as nx

from Island import Island
from Member import Member
from utils.save import path_decorator

from typing import List

import os


class Analyzer:
    def __init__(
        self,
        island: Island,
    ):
        self.island = island

    def generate_clear_graph(self) -> nx.DiGraph:

        self.clear_graph = nx.DiGraph()

        for member in self.island.current_members:
            self.clear_graph.add_node(member.surviver_id)
            self.island._get_neighbors(member)

            for target in member.current_clear_list:
                self.clear_graph.add_edge(member.surviver_id, target.surviver_id)

        return self.clear_graph

    def member_exist(self, member: Member) -> bool:
        if member in self.island.current_members:
            return True
        else: 
            return False


    def generate_network_by_decision(self):
        """
        
        """
        pass


    def connectivity(self) -> float:
        """
        计算成员之间的连接度
        """
        pass


class Tracer:
    def __init__(
        self,
        island_list: List[Island],
    ):
        sorted_island_list = sorted(
            island_list,
            key = lambda island: island.current_round,
        )

        self.analyzer_list = [Analyzer(island) for island in sorted_island_list]

    @classmethod
    def load_from_pickle_folder(
        cls,
        path: str,
        lower_round: float = -np.inf,
        upper_round: float = np.inf,
    ) -> "Tracer":
        """
        文件名都是 整数.pkl
        """
        path = path_decorator(path)
        files = os.listdir(path)

        island_list = []
        for file in files:
            round_idx, _ = file.split(".")
            if int(round_idx) >= lower_round and int(round_idx) <= upper_round:
                island = Island.load_from_pickle(path + file)
                island_list.append(island)

        return cls(island_list)


    def relevant_episode(self, member: Member) -> "Tracer":
        island_list = []
        for analyzer in self.analyzer_list:
            if analyzer.member_exist(member):
                island_list.append(analyzer.island)
        return Tracer(island_list)


