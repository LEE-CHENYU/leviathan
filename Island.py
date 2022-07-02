import numpy as np
from Member import Member
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

class Island():
    MIN_VICTIM_MEMORY, MAX_VICTIM_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0
    MIN_BENEFIT_MEMORY, MAX_BENEFIT_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0

    def __init__(
        self, 
        init_member_number: int
    ) -> None:

        # 初始人数，当前人数
        self._NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # 初始人物列表，全体人物列表，当前人物列表
        self.init_members = [Member(self._NAME_LIST[i], id=i, surviver_id=i) for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # 初始人物关系
        # 关系矩阵M，第j行 (M[j, :]) 代表第j个主体的记忆（受伤/受赠……）
        self.victim_memory = np.random.uniform(Island.MIN_VICTIM_MEMORY, Island.MAX_VICTIM_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.victim_memory[self.victim_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.victim_memory, np.nan)

        self.benefit_memory = np.random.uniform(Island.MIN_BENEFIT_MEMORY, Island.MAX_BENEFIT_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.benefit_memory[self.benefit_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.benefit_memory, np.nan)

        self._relationships = [self.victim_memory, self.benefit_memory]



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

        

    def relationship_modify(self):
        """保证自身nan"""
        pass

    def save_current_island(self):
        pass

