import numpy as np
from Member import Member
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from time import time

class Island():
    _MIN_VICTIM_MEMORY, _MAX_VICTIM_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0
    _MIN_BENEFIT_MEMORY, _MAX_BENEFIT_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0

    def __init__(
        self, 
        init_member_number: int,
        random_seed: float = None
    ) -> None:

        # 设置并记录随机数种子
        if random_seed is not None:
            self._random_seed = random_seed
        else:
            self._random_seed = time()
        self._rng = np.random.default_rng(self._random_seed)

        # 初始人数，当前人数
        self._NAME_LIST = self._rng.permutation(np.loadtxt("./name_list.txt", dtype=str))

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # 初始人物列表，全体人物列表，当前人物列表
        self.init_members = [Member(self._NAME_LIST[i], id=i, surviver_id=i, rng=self._rng) for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # 初始人物关系
        # 关系矩阵M，第j行 (M[j, :]) 代表第j个主体的被动记忆（受伤/受赠……）
        # 若要修改（增减）人物关系，需要修改：self.XXX_memory, self._relationships, Member.DECISION_INPUT_NAMES, Member._generate_decision_inputs()
        self.victim_memory = self._rng.uniform(Island._MIN_VICTIM_MEMORY, Island._MAX_VICTIM_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.victim_memory[self.victim_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.victim_memory, np.nan)

        self.benefit_memory = self._rng.uniform(Island._MIN_BENEFIT_MEMORY, Island._MAX_BENEFIT_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.benefit_memory[self.benefit_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.benefit_memory, np.nan)

        self._relationships = [self.victim_memory, self.benefit_memory]
        assert len(self._relationships) == Member._RELATION_SCALES, "关系矩阵数量和关系矩阵缩放量数量不一致"

    #########################################################################
    ################################ 基本操作 ################################# 

    def _backup_member_list(
        self, 
        member_list: List[Member]
    ) -> List[Member]:
        """复制member_list"""
        return [member for member in member_list]

    def _member_list_append(
        self, 
        append: List[Member] = [], 
        appended_rela_rows: np.ndarray = [], 
        appended_rela_columnes: np.ndarray = []
    ) -> None:
        """
        向current_members，all_members增加人物，
        增加current_member_num，
        修改relationships矩阵，
        修改人物surviver_id
        """
        appended_num = len(append)
        prev_member_num = self.current_member_num
 
        assert appended_rela_columnes.shape == (prev_member_num, appended_num), "输入关系列形状不匹配"
        assert appended_rela_rows.shape == (appended_num, prev_member_num), "输入关系行形状不匹配"

        # 向列表中增加人物
        for member in append:
            member.surviver_id = self.current_member_num
            self.current_members.append(member)
            self.all_members.append(member)

            self.current_member_num += 1

        # 修改关系矩阵
        for idx in range(len(self._relationships)):
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp_old = self._relationships[idx].copy()
            tmp_new = np.zeros((self.current_member_num, self.current_member_num))
            
            tmp_new[:prev_member_num, :prev_member_num] = tmp_old
            tmp_new[:prev_member_num, prev_member_num:] = appended_rela_columnes
            tmp_new[prev_member_num:, :prev_member_num] = appended_rela_rows
            np.fill_diagonal(tmp_new, np.nan)

            self._relationships[idx].resize(
                (self.current_member_num, self.current_member_num), 
                refcheck=False
            )
            self._relationships[idx][:] = tmp_new          # 仅修改数组内容，不修改列表里的数组地址

        return

    def _member_list_drop(
        self, 
        drop: List[Member] = []
    ) -> None:
        """
        从current_members删除人物，
        减少current_member_num，
        修改relationships矩阵，
        重新修改全体人物surviver_id
        """
        drop_id = np.array([member.id for member in drop])            # 校对id，确保正确删除
        drop_sur_id = np.array([member.surviver_id for member in drop])

        assert not (drop_sur_id == None).any(), "被删除对象没有surviver_id"

        # 排序，确保正确移除
        argsort_sur_id = np.argsort(drop_sur_id)[::-1]
        drop_id = drop_id[argsort_sur_id]
        drop_sur_id = drop_sur_id[argsort_sur_id]
        
        # 从列表中移除人物
        for idx in range(len(drop_id)):
            id_to_drop = drop_id[idx]
            sur_id_to_drop = drop_sur_id[idx]
            assert self.current_members[sur_id_to_drop].id == id_to_drop, "删除对象不匹配"

            self.current_members[sur_id_to_drop].surviver_id = None

            del self.current_members[sur_id_to_drop]
            self.current_member_num -= 1

        # 修改关系矩阵
        for idx in range(len(self._relationships)):
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp = np.delete(self._relationships[idx], drop_sur_id, axis=0)
            tmp = np.delete(tmp, drop_sur_id, axis=1)
            
            self._relationships[idx].resize(
                (self.current_member_num, self.current_member_num), 
                refcheck=False
            )
            self._relationships[idx][:] = tmp    # 仅修改数组内容，不修改列表里的数组地址

        # 重新编号存活成员
        for sur_id in range(self.current_member_num):
            self.current_members[sur_id].surviver_id = sur_id
        
        return

    def member_list_modify(
        self, 
        append: List[Member] = [], 
        drop: List[Member] = [], 
        appended_rela_rows: np.ndarray = [], 
        appended_rela_columnes: np.ndarray = []
    ) -> None:
        """修改member_list，先增加人物，后修改""" 
        self._member_list_append(
            append=append, 
            appended_rela_rows=appended_rela_rows, appended_rela_columnes=appended_rela_columnes
        )
        self._member_list_drop(
            drop=drop
        )

        return

    def _overlap_of_relations(
        self, 
        principal: Member, 
        object: Member
        ) -> List[float]:
        """计算关系网内积"""

        def normalize(arr):
            norm = np.linalg.norm(arr)
            if norm == 0:
                return 0
            else:
                return arr / norm

        overlaps = []
        for relationship in self._relationships:
            pri_row = normalize(relationship[principal.surviver_id, :])
            pri_col = normalize(relationship[:, principal.surviver_id])
            obj_row = normalize(relationship[object.surviver_id, :])
            obj_col = normalize(relationship[:, object.surviver_id])

            overlaps.append((
                np.sum(pri_row * obj_row)
                + np.sum(pri_row * obj_col)
                + np.sum(pri_col * obj_row)
                + np.sum(pri_col * obj_col)) / 4
            )
        
        return overlaps
    
    def _relations_w_normalize(
        self,
        principal: Member,
        object: Member
    ) -> List[float]:
        """计算归一化（tanh）后的关系矩阵元"""
        elements = []
        for relationship in self._relationships:
            elements.append(relationship[principal.surviver_id, object.surviver_id])
            elements.append(relationship[object.surviver_id, principal.surviver_id])

        elements = np.array(elements)
        return np.tanh(elements * Member._RELATION_SCALES)

    def relationship_modify(self):
        """保证自身nan"""
        pass    

    def _split_into_groups(
        self, 
        group_size: int = 10, 
        prob_group_in_action: float = 1.0
        ) -> List[List[Member]]:

        shuffled_members = self._backup_member_list(self.current_members)
        self._rng.shuffle(shuffled_members)

        idx_list_list = np.array_split(np.arange(self.current_member_num, dtype=int), np.round(self.current_member_num / group_size).astype(int))

        group_list = []
        for idx_list in idx_list_list:
            group = []
            for member_idx in idx_list:
                group.append(self.current_members[member_idx])
            
            # 每组按概率发生战斗
            if self._rng.random() < prob_group_in_action:
                group_list.append(group)

        return group_list

    def _get_pairs_from_group(self, principal, object):
        pass


    def save_current_island(self):
        pass

    #########################################################################
    ################################## 模拟 ################################## 
    def produce(self):
        for member in self.current_members:
            member.produce()

    def fight(
        self, 
        group_size: int = 10,
        prob_group_in_fight: float = 1.0
        ):

        # 打乱顺序，随机分组
        group_list = self._split_into_groups(group_size, prob_group_in_fight)
        print(group_list)


        

        

                
