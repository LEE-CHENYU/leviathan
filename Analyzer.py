import numpy as np 
import pandas as pd
import networkx as nx

from Island import Island
from Member import Member
from utils.save import path_decorator

from typing import List, Dict

import os


class Analyzer:
    def __init__(
        self,
        island: Island,
    ):
        self.island = island

    # Relationships
    # ##########################################################################
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

    def connectivity(self) -> float:
        """
        计算成员之间的连接度
        """
        pass

    # Member
    # ##########################################################################
    def look_for_current_member(self, member: Member | int) -> Member:
        if isinstance(member, Member):
            id = member.id
        elif isinstance(member, int): 
            id = member


        for target in self.island.current_members:
            if id == target.id:
                return target
            
        return None

    def member_exist(self, member: Member | int) -> bool:
        if self.look_for_current_member(member) is not None:
            return True
        else:
            return False
        
    def member_row(self, member: Member | int) -> pd.DataFrame:
        member = self.look_for_current_member(member)
        return member.save_to_row()

    def all_member_info(self) -> pd.DataFrame:
        current_member_df = self.member_row(self.island.current_members[0])
        
        for member in self.island.current_members[1:]:
            current_member_df = pd.concat([
                current_member_df,
                self.member_row(member)],
                axis=0
            )

        return current_member_df

    def round_info(self) -> Dict[str, pd.DataFrame]:
        info = pd.DataFrame({
            "round": self.island.current_round,
            "population": self.island.current_member_num,
        })

        df = self.all_member_info()
        return {
            "info": info,
            "mean": pd.DataFrame(df.mean(0, numeric_only=True)).transpose(),
            "std": pd.DataFrame(df.std(0, numeric_only=True)).transpose(),
            "quartile 1/4": pd.DataFrame(df.quantile(0.25, 0, numeric_only=True)).transpose(),
            "quartile 2/4": pd.DataFrame(df.quantile(0.5, 0, numeric_only=True)).transpose(),
            "quartile 3/4": pd.DataFrame(df.quantile(0.75, 0, numeric_only=True)).transpose(),
        }

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

    def relevant_episode(self, member: Member | int) -> "Tracer":
        island_list = []
        for analyzer in self.analyzer_list:
            if analyzer.member_exist(member):
                island_list.append(analyzer.island)
        return Tracer(island_list)

    def member_history(self, member: Member | int) -> pd.DataFrame:

        current_member_df = self.analyzer_list[0].member_row(member)
        
        for analyzer in self.analyzer_list[1:]:
            current_member_df = pd.concat(
                [
                    current_member_df,
                    analyzer.member_row(member)
                ],
                axis=0
            )

        return current_member_df

    def all_member_summary_history(self,) -> Dict[str, pd.DataFrame]:
        all_info_dict = self.analyzer_list[0].round_info()

        for analyzer in self.analyzer_list[1:]:
            new_info_dict = analyzer.round_info()

            for key, df in all_info_dict.items():
                df = pd.concat(
                    [
                        df,
                        new_info_dict[key],
                    ],
                    axis = 0
                )

                all_info_dict[key] = df

        index = np.array([ana.island.current_round for ana in self.analyzer_list])
        for key, df in all_info_dict.items():
            df = df.set_index(index)
            all_info_dict[key] = df

        return all_info_dict

    def surviver_cohort_matrix(
        self, 
        starting_idx: int,
        ending_idx: int,
    ) -> np.ndarray:

        pass