import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch

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
    
    def param_transform(self) -> np.array:
        """
        将本轮存活玩家的五个决策参数向量合并为一个巨型向量
        """
        parameter_dict_list = []

        for member in self.island.current_members:
            parameter_dict_list.append(member.parameter_dict)
        parameter_array = np.array([list(param_dict.values()) for param_dict in parameter_dict_list])
        self.monster_array = np.concatenate(np.transpose(parameter_array, axes = [1,0,2]), axis=1)
    
    def k_means_cluster(self, n_clusters=3, elbow = False):
        """
        在将本轮的参数向量中寻找聚类
        """
        self.param_transform()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters, random_state=0)
        self.clusters = kmeans.fit(self.monster_array)
        
        self.ranked_label_value_match = []
        for j in range(0,n_clusters):
            label_ranking = np.argsort(self.clusters.cluster_centers_, axis=1)
            param_label_list = ['attack_self_productivity', 'attack_self_vitality', 'attack_self_cargo', 'attack_self_age', 'attack_self_neighbor', 'attack_obj_productivity', 'attack_obj_vitality', 'attack_obj_cargo', 'attack_obj_age', 'attack_obj_neighbor', 'attack_victim_overlap', 'attack_benefit_overlap', 'attack_benefit_land_overlap', 'attack_victim_passive', 'attack_victim_active', 'attack_benefit_passive', 'attack_benefit_active', 'attack_benefit_land_passive', 'attack_benefit_land_active', 'offer_self_productivity', 'offer_self_vitality', 'offer_self_cargo', 'offer_self_age', 'offer_self_neighbor', 'offer_obj_productivity', 'offer_obj_vitality', 'offer_obj_cargo', 'offer_obj_age', 'offer_obj_neighbor', 'offer_victim_overlap', 'offer_benefit_overlap', 'offer_benefit_land_overlap', 'offer_victim_passive', 'offer_victim_active', 'offer_benefit_passive', 'offer_benefit_active', 'offer_benefit_land_passive', 'offer_benefit_land_active', 'reproduce_self_productivity', 'reproduce_self_vitality', 'reproduce_self_cargo', 'reproduce_self_age', 'reproduce_self_neighbor', 'reproduce_obj_productivity', 'reproduce_obj_vitality', 'reproduce_obj_cargo', 'reproduce_obj_age', 'reproduce_obj_neighbor', 'reproduce_victim_overlap', 'reproduce_benefit_overlap', 'reproduce_benefit_land_overlap', 'reproduce_victim_passive', 'reproduce_victim_active', 'reproduce_benefit_passive', 'reproduce_benefit_active', 'reproduce_benefit_land_passive', 'reproduce_benefit_land_active', 'clear_self_productivity', 'clear_self_vitality', 'clear_self_cargo', 'clear_self_age', 'clear_self_neighbor', 'clear_obj_productivity', 'clear_obj_vitality', 'clear_obj_cargo', 'clear_obj_age', 'clear_obj_neighbor', 'clear_victim_overlap', 'clear_benefit_overlap', 'clear_benefit_land_overlap', 'clear_victim_passive', 'clear_victim_active', 'clear_benefit_passive', 'clear_benefit_active', 'clear_benefit_land_passive', 'clear_benefit_land_active', 'offer_land_self_productivity', 'offer_land_self_vitality', 'offer_land_self_cargo', 'offer_land_self_age', 'offer_land_self_neighbor', 'offer_land_obj_productivity', 'offer_land_obj_vitality', 'offer_land_obj_cargo', 'offer_land_obj_age', 'offer_land_obj_neighbor', 'offer_land_victim_overlap', 'offer_land_benefit_overlap', 'offer_land_benefit_land_overlap', 'offer_land_victim_passive', 'offer_land_victim_active', 'offer_land_benefit_passive', 'offer_land_benefit_active', 'offer_land_benefit_land_passive', 'offer_land_benefit_land_active']
            value_ranked = np.sort(self.clusters.cluster_centers_, axis=1)
            param_label_list_ranked = [param_label_list[i] for i in label_ranking[j]]
            self.ranked_label_value_match.append(list(zip(param_label_list_ranked, value_ranked[j])))
            
        def elbow_method(self):
                # Calculate the within-cluster sum of squares (WCSS) for different values of k
                wcss = []
                for k in range(1, 20):
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    kmeans.fit(self.monster_array)
                    wcss.append(kmeans.inertia_)

                # Plot the WCSS against k
                plt.plot(range(1, 50), wcss)
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
                plt.title('Elbow Method')
                plt.show()
            
        if elbow == True:
            elbow_method()
                  
    def heir_cluster(self):
        """
        在将本轮的参数向量中寻找聚类
        """
        # Compute the linkage matrix
        Z = sch.linkage(self.monster_array, method='ward')

        # Plot the dendrogram
        plt.figure(figsize=(10, 6))
        sch.dendrogram(Z)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.title('Dendrogram')
        plt.show()
    
    def pca(self, three_d = False, cluster = False, n_clusters=3):
        """
        将本轮的参数向量映射在PCA平面上
        """
        
        self.param_transform()
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.monster_array)

        if cluster == True:
            self.k_means_cluster(n_clusters)
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.clusters.labels_)
        # Plot the mapped data
        elif cluster == False:
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Mapping onto a 2-D Space')
        plt.show()
        
        if three_d == True:
            pca_3d = PCA(n_components=3)
            pca_3d_result = pca_3d.fit_transform(self.monster_array)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2])
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

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