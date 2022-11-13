import numpy as np 
import networkx as nx

from Island import Island


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


    def generate_network_by_decision(self):
        """
        
        """
        pass


    def connectivity(self):
        """
        计算成员之间的连接度
        """
        pass


class Tracer:
    pass

