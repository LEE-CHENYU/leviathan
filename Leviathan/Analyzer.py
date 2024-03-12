import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.decomposition import PCA

from Leviathan.Island import Island
from Leviathan.Member import Member
from utils.save import path_decorator

from typing import List, Dict, Union

import os

class Analyzer:
    
    ## single-round analysis
    def __init__(
        self,
        island: Island,
    ):
        self.island = island

    # Relationships
    # ##########################################################################
    def clear_graph(self) -> nx.DiGraph:

        clear_graph = nx.DiGraph()

        for member in self.island.current_members:
            clear_graph.add_node(member.surviver_id)
            self.island._get_neighbors(member, backend="inner product")

            for target in member.current_clear_list:
                clear_graph.add_edge(member.surviver_id, target.surviver_id)

        return clear_graph

    def generate_network_by_decision(self):
        """
        
        """
        pass

    def clear_degree(self) -> float:
        """
        计算成员之间通行权的连接度
        """
        return self.clear_graph().degree()
    
    def pca_analysis(self):
        """
        将本轮的参数向量映射在PCA平面上
        """

        # Perform PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(parameter_array)

        # Plot the mapped data
        plt.scatter(data_2d[:, 0], data_2d[:, 1])
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Mapping onto a 2-D Space')
        plt.show()

    # Member
    # ##########################################################################
    def look_for_current_member(self, member: Member | int) -> Union[Member, None]:
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
            "round": [self.island.current_round],
            "population": [self.island.current_member_num],
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
    
    ## multi-round analysis
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

            try:
                round_idx = int(round_idx)
            except:
                continue

            if round_idx >= lower_round and round_idx <= upper_round:
                island = Island.load_from_pickle(path + file)
                island_list.append(island)

        return cls(island_list)

    def relevant_episode(self, member: Member | int) -> "Tracer":
        island_list = []
        for analyzer in self.analyzer_list:
            if analyzer.member_exist(member):
                island_list.append(analyzer.island)
        return Tracer(island_list)
    
    def analyzer_by_round(self, round: int) -> Analyzer:
        for analyzer in self.analyzer_list:
            if analyzer.island.current_round == round:
                return analyzer

    # history ==========================================================
    def member_history(self, member: Member | int) -> pd.DataFrame:
        """
        从所有的岛屿中找到这个成员的历史
        """
        for idx, analyzer in enumerate(self.analyzer_list):
            if analyzer.member_exist(member):
                break

        current_member_df = self.analyzer_list[idx].member_row(member)
        
        for analyzer in self.analyzer_list[idx+1:]:
            if not analyzer.member_exist(member):
                continue

            current_member_df = pd.concat(
                [
                    current_member_df,
                    analyzer.member_row(member)
                ],
                axis=0
            )

        return current_member_df

    def all_member_summary_history(self,) -> Dict[str, pd.DataFrame]:
        """
        从所有的岛屿中找到所有成员的历史
        """
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

    # land =============================================================
    def land_animation(self, show_id: bool = False):
        fig, axs = plt.subplots(1, 2, figsize=(10, 7))
        self.analyzer_list[0].island.land.plot(axs, show_id=show_id)

        fig.subplots_adjust(bottom=0.25)

        ax_round = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        round_slider = Slider(
            ax=ax_round,
            label='round',
            valmin=self.analyzer_list[0].island.current_round,
            valmax=self.analyzer_list[-1].island.current_round,
            valinit=self.analyzer_list[0].island.current_round,
            valstep=self.analyzer_list[1].island.current_round - self.analyzer_list[0].island.current_round 
        )

        def update(val):
            # remove the previous plot
            axs[0].cla()
            axs[1].cla()

            # plot the new plot
            round = int(round_slider.val)
            analyzer = self.analyzer_by_round(round)
            analyzer.island.land.plot(axs, show_id=show_id)
            fig.canvas.draw_idle()

        round_slider.on_changed(update)

        plt.show()