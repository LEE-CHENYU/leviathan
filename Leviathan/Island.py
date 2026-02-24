import math
import numpy as np
import pandas as pd
from Leviathan.Member import Member, colored
from Leviathan.Land import Land
from Leviathan.settings import name_list

from utils.save import path_decorator

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import time
import logging
import pickle 
# import dill
import sys
import itertools

from collections import defaultdict

def _requirement_for_reproduction(
    member_1: Member, 
    member_2: Member
) -> bool:
    result = (
        (
            member_1.vitality + member_1.cargo
            + member_2.vitality + member_2.cargo
        ) >= Island._REPRODUCE_REQUIREMENT 
        and 
        member_1.is_qualified_to_reproduce
        and 
        member_2.is_qualified_to_reproduce
    )

    # 展示无法生育的原因 （for debugging）
    # if not result:
    #     if (
    #         member_1.vitality + member_1.cargo
    #         + member_2.vitality + member_2.cargo
    #     ) < Island._REPRODUCE_REQUIREMENT :
    #         reason = "Not enough vitality and cargo"

    #     elif not member_1.is_qualified_to_reproduce:
    #         reason = f"{member_1} not qualified to reproduce"

    #     elif not member_2.is_qualified_to_reproduce:
    #         reason = f"{member_2} not qualified to reproduce"

    #     print(f"{member_1} and {member_2} cannot reproduce because: {reason}")

    return result

def _requirement_for_offer(
    member_1: Member, 
    member_2: Member
) -> bool:
    return member_1.cargo * Member._MIN_OFFER_PERCENTAGE >= Member._MIN_OFFER

def _requirement_for_offer_land(
    member_1: Member, 
    member_2: Member
) -> bool:
    return member_1.land_num > 1

class Island():
    _MIN_MAX_INIT_RELATION = {
        "victim": [-500, 1000],                # 若随机到负值，则该记忆初始化为0
        "benefit": [-500, 1000],        
        "benefit_land": [-30, 30],           
    }

    _NEIGHBOR_SEARCH_RANGE = 1000
    _REPRODUCE_REQUIREMENT = 100                            # 生育条件：双亲血量和仓库之和大于这个值
    assert _REPRODUCE_REQUIREMENT > Member._CHILD_VITALITY

    # 记录/输出周期
    _RECORD_PERIOD = 1

    def __init__(
        self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
    ) -> None:

        # 设置并记录随机数种子
        self._create_from_file = False
        self._file_name = ""
        if random_seed is not None:
            self._random_seed = int(random_seed)
        else:
            self._random_seed = int(time())
        self._rng = np.random.default_rng(self._random_seed)

        # 保存路径
        self._save_path = path_decorator(save_path)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'{save_path}/log.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # 初始人数，当前人数
        self._NAME_LIST = self._rng.permutation(name_list)

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # 初始人物列表，全体人物列表，当前人物列表
        self.init_members = [Member(self._NAME_LIST[i], id=i, surviver_id=i, rng=self._rng) for i in range(self.init_member_num)]
        print(f"Init members: {self.init_members[3].surviver_id}")
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # 地图
        assert land_shape[0] * land_shape[1] > init_member_number, "土地面积应该大于初始人口"
        self.land = Land(land_shape)
        # 为初始人口分配土地
        _loc_idx_list = self._rng.choice(
            a = range(land_shape[0] * land_shape[1]),
            size = self.init_member_num,
            replace = False,
        )
        _loc_list = [(int(loc_idx / land_shape[0]), loc_idx % land_shape[0]) for loc_idx in _loc_idx_list]
        for idx in range(self.init_member_num):
            self._acquire_land(self.all_members[idx], _loc_list[idx])

        # 初始人物关系
        # 关系矩阵M，第j行 (M[j, :]) 代表第j个主体的被动记忆（受伤/受赠……）
        # 若要修改（增减）人物关系，需要修改：self.relationship_dict, Member.DECISION_INPUT_NAMES, Member._generate_decision_inputs()
        self.relationship_dict = {
            'victim': None, 
            'benefit': None,
            'benefit_land': None
        }

        # Then populate with random values as before
        for key in self.relationship_dict:
            min_val, max_val = Island._MIN_MAX_INIT_RELATION[key]
            rela = self._rng.uniform(min_val, max_val, size=(self.init_member_num, self.init_member_num))
            np.fill_diagonal(rela, np.nan)
            self.relationship_dict[key] = rela

        assert len(self.relationship_dict) == len(Member._RELATION_SCALES), "关系矩阵数量和关系矩阵缩放量数量不一致"

        # 记录动作 （每Island._RECORD_PERIOD输出、清空一次）
        self.record_action_dict = {
            "attack": {},
            "benefit": {},
            "benefit_land": {},
            "reproduce": {},
            "clear": {},
        }
        self.round_action_dict = {
            "attack": {},
            "benefit": {},
            "benefit_land": {},
            "reproduce": {},
            "clear": {},
        }
        self.record_born = []
        self.record_death = []

        # 记录状态 （每Island._RECORD_PERIOD向末尾增append一个0）
        self.record_total_production = [0]
        self.record_total_consumption = [0]
        self.round_total_production = 0.0
        self.round_total_consumption = 0.0
        self.record_total_dict = {
            "attack": [0],
            "benefit": [0],
            "benefit_land": [0],
            "reproduce": [0],
            "clear": [0],
        }
        self.round_total_dict = {
            "attack": 0.0,
            "benefit": 0.0,
            "benefit_land": 0.0,
            "reproduce": 0.0,
            "clear": 0.0,
        }
        self.record_historic_ratio_list = np.array([(0,0,0,0)])
        self.record_historic_ranking_list = [(0,0,0)]
        self.record_payoff_matrix = np.zeros((8, 64))
        self.record_land = [self.land.owner_id]
        
        self.previous_vitalities = {}
        self.vitality_diff = {}

        # 回合数
        self.current_round = 0

        # 行为学习与环境上下文
        self.round_context = {}
        self._round_snapshot = {
            member.id: (member.vitality, member.cargo, member.land_num)
            for member in self.current_members
        }


    ############################################################################
    ################################ 基本操作 #################################### 

    def member_by_name(
        self,
        name: str,
    ) -> Member:
        for member in self.current_members:
            if member.name == name:
                return member
        
        for member in self.all_members:
            if member.name == name:
                return member

        raise KeyError(f"Member {name} not found!")

    # =============================== 成员增减 ===================================
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
        appended_rela_columns: np.ndarray = [],
    ) -> None:
        """
        向current_members，all_members增加一个列表的人物，
        增加current_member_num，
        修改relationships矩阵，
        修改人物surviver_id，
        """
        appended_num = len(append)
        prev_member_num = self.current_member_num

        if not isinstance(appended_rela_columns, np.ndarray):
            raise ValueError("关系矩阵增添的列应该是ndarray类型")
        if not isinstance(appended_rela_rows, np.ndarray):
            raise ValueError("关系矩阵增添的行应该是ndarray类型")
        assert appended_rela_columns.shape == (prev_member_num, appended_num), "输入关系列形状不匹配"
        assert appended_rela_rows.shape == (appended_num, prev_member_num), "输入关系行形状不匹配"

        # 向列表中增加人物
        for member in append:
            member.surviver_id = self.current_member_num
            self.current_members.append(member)
            self.all_members.append(member)

            self.current_member_num += 1

        # 记录出生
        self.record_born = self.record_born + append

        # 修改关系矩阵
        for key in self.relationship_dict.keys():
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp_old = self.relationship_dict[key].copy()
            tmp_new = np.zeros((self.current_member_num, self.current_member_num))
            
            tmp_new[:prev_member_num, :prev_member_num] = tmp_old
            tmp_new[:prev_member_num, prev_member_num:] = appended_rela_columns
            tmp_new[prev_member_num:, :prev_member_num] = appended_rela_rows
            np.fill_diagonal(tmp_new, np.nan)

            self.relationship_dict[key] = tmp_new

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
        print(f"Drop {len(drop)} members", drop)
        drop_id = np.array([member.id for member in drop])            # 校对id，确保正确删除
        print(f"Drop IDs: {drop_id}")
        drop_sur_id = np.array([member.surviver_id for member in drop])
        print(f"Drop Survivor IDs: {drop_sur_id}")

        if (drop_sur_id == None).any():
            raise AttributeError(f"被删除对象应该有surviver_id")

        for member in drop:
            assert member.owned_land == [], "被删除对象应该没有土地"

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
        for key in self.relationship_dict.keys():
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp = np.delete(self.relationship_dict[key], drop_sur_id, axis=0)
            tmp = np.delete(tmp, drop_sur_id, axis=1)

            self.relationship_dict[key] = tmp

        # 重新编号存活成员
        for sur_id in range(self.current_member_num):
            self.current_members[sur_id].surviver_id = sur_id
        
        return

    def member_list_modify(
        self, 
        append: List[Member] = [], 
        drop: List[Member] = [], 
        appended_rela_rows: np.ndarray = np.empty(0), 
        appended_rela_columns: np.ndarray = np.empty(0)
    ) -> None:
        """
        修改member_list，先增加人物，后修改
        记录出生、死亡
        """
        if append != []:
            self._member_list_append(
                append=append, 
                appended_rela_rows=appended_rela_rows, appended_rela_columns=appended_rela_columns
            )
        if drop != []:
            self._member_list_drop(
                drop=drop
            )

        return

    @property
    def is_dead(self,) -> bool:
        return self.current_member_num == 0

# ================================ 关系矩阵修改 ==================================
    def _overlap_of_relations(
        self, 
        principal: Member, 
        object: Member,
        normalize: bool = True,
        ) -> List[float]:
        """计算关系网内积"""

        def _normalize(arr):
            """剔除nan，归一化向量"""
            arr[principal.surviver_id] = 0
            arr[object.surviver_id] = 0
            norm = np.linalg.norm(arr)
            if norm == 0:
                return 0
            else:
                if normalize:
                    return arr / norm
                else:
                    return arr

        overlaps = []
        for relationship in list(self.relationship_dict.values()):
            
            pri_row = _normalize(relationship[principal.surviver_id, :].copy())
            pri_col = _normalize(relationship[:, principal.surviver_id].copy())
            obj_row = _normalize(relationship[object.surviver_id, :].copy())
            obj_col = _normalize(relationship[:, object.surviver_id].copy())

            overlaps.append((
                np.sum(np.sqrt(np.maximum(pri_row * obj_row, 0)))
                + np.sum(np.sqrt(np.maximum(pri_row * obj_col, 0)))
                + np.sum(np.sqrt(np.maximum(pri_col * obj_row, 0)))
                + np.sum(np.sqrt(np.maximum(pri_col * obj_col, 0))) / 4
            ))
        
        return overlaps

    def _relations_w_normalize(
        self,
        principal: Member,
        object: Member,
        normalize: bool = True,
    ) -> np.ndarray:
        """计算归一化（tanh）后的关系矩阵元"""
        elements = []
        for relationship in list(self.relationship_dict.values()):
            elements.append(relationship[principal.surviver_id, object.surviver_id])
            elements.append(relationship[object.surviver_id, principal.surviver_id])

        elements = np.array(elements)
        
        if normalize:
            result = elements * np.repeat(Member._RELATION_SCALES, 2)
            return np.tanh(result)
        else:
            return elements

    def relationship_modify(
        self, 
        relationship_name,
        member_1: Member, 
        member_2: Member, 
        add_value: float
    ) -> None:
        """
        增加矩阵元[member_1.surviver_id, member_2.surviver_id]
        """
        assert member_1 is not member_2, "不能修改关系矩阵中的对角元素"
        # Add key existence check
        if relationship_name not in self.relationship_dict:
            self.relationship_dict[relationship_name] = np.full(
                (self.init_member_num, self.init_member_num), np.nan
            )
        
        relationship = self.relationship_dict[relationship_name]
        relationship[member_1.surviver_id, member_2.surviver_id] += add_value

# =================================== 土地 ======================================
    def _acquire_land(
        self, 
        member: Member, 
        location: Tuple[int, int],
    ) -> None:
        assert self.land[location] is None, "获取的土地应该没有主人"

        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = member
        member.acquire_land(location)

    def _discard_land(
        self,
        member: Member,
        location: Tuple[int, int],
    ) -> None:
        assert location in member.owned_land, "只能丢弃拥有的土地"
        assert self.land[location] == member, "只能丢弃自己的土地"

        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = None
        member.discard_land(location)

    def get_neighbors(self):
        for member in self.current_members:
            self._get_neighbors(member, backend="inner product")

    def _get_neighbors(
        self, 
        member: Member,
        backend: Optional[str] = None,
    ) -> None:
        """
        存储四个列表：
        - clear_list: 允许通行
        - self_blocked_list: 与member直接接壤
        - neighbor_blocked_list: 与member间接接壤的成员以及作为桥梁的地主，存储格式为（地主，间接接壤成员）
        - empty_loc_list: 闲置土地
        """
        (
            member.current_clear_list,
            member.current_self_blocked_list,
            member.current_neighbor_blocked_list,
            member.current_empty_loc_list,
        ) = self.land.neighbors(
            member, 
            self, 
            Island._NEIGHBOR_SEARCH_RANGE,
            # decision_threshold=1,
            backend,
        )

    def _maintain_neighbor_list(self, member: Member):
        """
        删除已经死亡的邻居
        """
        member.current_clear_list = [
            neighbor for neighbor in member.current_clear_list 
            if not self.all_members[neighbor].autopsy()
        ]
        member.current_self_blocked_list = [
            neighbor for neighbor in member.current_self_blocked_list 
            if not self.all_members[neighbor].autopsy()
        ]
        member.current_neighbor_blocked_list = [
            neighbor for neighbor in member.current_neighbor_blocked_list 
            if not self.all_members[neighbor[0]].autopsy() and not self.all_members[neighbor[1]].autopsy()
        ]
        member.current_empty_loc_list = [
            loc for loc in member.current_empty_loc_list
            if self.land[loc] is None
        ]

    def resolve_member_index(
        self,
        member_ref: Union[int, Member],
        prefer_index: bool = True,
    ) -> Optional[int]:
        """
        Resolve a member reference to the current_members index.

        - If prefer_index is True, integers are treated as current_members indices.
        - If prefer_index is False, integers are treated as stable member.id values.
        - Member objects are resolved via survivier_id when available.
        """
        if member_ref is None:
            return None

        if isinstance(member_ref, Member):
            idx = getattr(member_ref, "surviver_id", None)
            if isinstance(idx, int) and 0 <= idx < len(self.current_members):
                return idx
            member_ref = getattr(member_ref, "id", None)
            if member_ref is None:
                return None

        try:
            ref_int = int(member_ref)
        except (TypeError, ValueError):
            return None

        if prefer_index and 0 <= ref_int < len(self.current_members):
            return ref_int

        for idx, member in enumerate(self.current_members):
            if getattr(member, "id", None) == ref_int:
                return idx

        if 0 <= ref_int < len(self.current_members):
            return ref_int
        return None

    def resolve_member_index_by_id(self, member_id: Union[int, Member]) -> Optional[int]:
        """Resolve a stable member.id to the current_members index."""
        return self.resolve_member_index(member_id, prefer_index=False)

    def resolve_member_id(self, member_ref: Union[int, Member]) -> Optional[int]:
        """Resolve a member reference to the stable member.id."""
        if isinstance(member_ref, Member):
            return getattr(member_ref, "id", None)
        idx = self.resolve_member_index(member_ref, prefer_index=True)
        if idx is None:
            return None
        if 0 <= idx < len(self.current_members):
            return getattr(self.current_members[idx], "id", None)
        return None

    def get_member_by_id(self, member_id: Union[int, Member]) -> Optional[Member]:
        """Return the current member object by stable member.id, if alive."""
        idx = self.resolve_member_index_by_id(member_id)
        if idx is None:
            return None
        return self.current_members[idx]

    def _find_targets(
        self,
        member: Member,
        target_list: List[Member],
        decision_name: str,
        prob_of_action: float = 1.0,
        other_requirements: Callable = None,
        bilateral: bool = False,
        land_owner_decision: str = "",
        backend: Optional[str] = None,
    ) -> List[Member]:
        """
        根据决策函数，从潜在对象列表中选出对象
        decision_name: Member.parameter_dict的keys之一
        other_requirements: 函数，输入为两个Member，输出True（通过要求）/False
        bilateral: 设为True后，决策函数的判定为双向符合
        landlord_decision: 地主的决策
        """

        if target_list == []:
            return []

        selected_target = []
        for tgt in target_list:

            if self._rng.uniform(0, 1) > prob_of_action:
                continue

            # 检查tgt是元组（包含地主）还是单一成员
            if isinstance(tgt, tuple):
                land_owner, obj = tgt
            elif isinstance(tgt, Member):
                obj = tgt
                land_owner = None
            else:
                raise ValueError("请在列表中输入正确的目标：成员、或（土地主人，成员）")

            # 检查是否重复判断obj
            if obj in selected_target:
                continue

            if other_requirements is not None:
                if not other_requirements(member, obj):
                    continue

            if not member.decision(
                decision_name,
                obj,
                self,
                backend=backend,
                logger=self._logger,
            ):  
                continue

            if bilateral:
                if not obj.decision(
                    decision_name,
                    member,
                    self,
                    backend=backend,
                    logger=self._logger,
                ):
                    continue

            if land_owner_decision != "" and land_owner is not None:
                if not land_owner.decision(
                    land_owner_decision,
                    obj,
                    self,
                    backend=backend,
                ):
                    continue

            selected_target.append(obj)

        return list(set(selected_target))

# ##############################################################################
# ##################################### 记录 ####################################
    def _record_actions(
        self, 
        record_name: str, 
        member_1: Member, 
        member_2: Member, 
        value_1: float, 
        value_2: float = None
    ):
        record_dict = self.record_action_dict[record_name]
        round_record_dict = self.round_action_dict[record_name]

        # 记录双方的动作
        for target_dict in (record_dict, round_record_dict):
            try:
                target_dict[(member_1.id, member_2.id)] += value_1
            except KeyError:
                target_dict[(member_1.id, member_2.id)] = value_1
        if value_2 is not None:
            for target_dict in (record_dict, round_record_dict):
                try:
                    target_dict[(member_2.id, member_1.id)] += value_2
                except KeyError:
                    target_dict[(member_2.id, member_1.id)] = value_2

        # 记录总动作
        if value_2 is not None:
            self.record_total_dict[record_name][-1] += value_1 + value_2
            self.round_total_dict[record_name] += value_1 + value_2
        else:
            self.record_total_dict[record_name][-1] += value_1
            self.round_total_dict[record_name] += value_1

    def _record_single_action(
        self,
        record_name: str,
        member: Member,
        value: float = 1.0,
    ) -> None:
        """Record a single-actor action without a target member."""
        record_dict = self.record_action_dict[record_name]
        round_record_dict = self.round_action_dict[record_name]
        key = (member.id, -1)

        for target_dict in (record_dict, round_record_dict):
            try:
                target_dict[key] += value
            except KeyError:
                target_dict[key] = value

        self.record_total_dict[record_name][-1] += value
        self.round_total_dict[record_name] += value
    
    def generate_decision_history(self) -> None:
        if not hasattr(self, 'decision_history'):
            self.decision_history = {}
            
        for member_1 in self.all_members: # possibly should be current_members
            if member_1.id not in self.decision_history:
                self.decision_history[member_1.id] = {}
            self.decision_history[member_1.id][self.current_round] = (0, 0, 0)
        
        for (member_1, member_2) in self.round_action_dict['attack']: 
            ## member_1 here is member_1.id
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (1, prev_decisions[1], prev_decisions[2])
        
        for (member_1, member_2) in self.round_action_dict['benefit']:
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (prev_decisions[0], 1, prev_decisions[2])
        
        for (member_1, member_2) in self.round_action_dict['benefit_land']:
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (prev_decisions[0], prev_decisions[1], 1)

    def record_historic_ratio(self):
        current_attack_ratio = self.record_total_dict['attack'][-1]/(self.current_member_num)
        current_benefit_ratio = self.record_total_dict['benefit'][-1]/(self.current_member_num)
        current_benefit_land_ratio = self.record_total_dict['benefit_land'][-1]/(self.current_member_num)
        # current_reproduce_ratio = self.record_total_dict[-1]['reproduce']/(self.current_member_num) #预留reproduce

        self.record_historic_ratio_list = np.append(self.record_historic_ratio_list, [[current_attack_ratio, current_benefit_ratio, current_benefit_land_ratio, 0]], axis=0)

    def record_historic_ranking(self):
        # 计算排位
        current_attack_ranking = (sorted(self.record_historic_ratio_list[:,0]).index(self.record_historic_ratio_list[:,0][-1]) + 1)/len(self.record_historic_ratio_list[:,0])
        current_benefit_ranking = (sorted(self.record_historic_ratio_list[:,1]).index(self.record_historic_ratio_list[:,1][-1]) + 1)/len(self.record_historic_ratio_list[:,1])
        current_benefit_land_ranking = (sorted(self.record_historic_ratio_list[:,2]).index(self.record_historic_ratio_list[:,2][-1]) + 1)/len(self.record_historic_ratio_list[:,2])
        self.record_historic_ranking_list.append((current_attack_ranking, current_benefit_ranking, current_benefit_land_ranking))

    def calculate_histoic_quartile(self):
        # Flatten the list of tuples
        flattened_list = [value for t in self.record_historic_ranking_list for value in t]

        # Sort the flattened list
        sorted_list = sorted(flattened_list)

        # Determine quartile boundaries
        q1_boundary = sorted_list[len(sorted_list) // 4]
        q2_boundary = sorted_list[len(sorted_list) // 2]
        q3_boundary = sorted_list[3 * len(sorted_list) // 4]

        def determine_quartile(value):
            if value <= q1_boundary:
                return 1
            elif value <= q2_boundary:
                return 2
            elif value <= q3_boundary:
                return 3
            else:
                return 4

        # Map each value in the tuples to its corresponding quartile
        # Use a dictionary with the round as the key and the quartile tuple as the value
        self.record_historic_quartile_dict = {current_round: (determine_quartile(val1), determine_quartile(val2), determine_quartile(val3)) 
                                            for current_round, (val1, val2, val3) in enumerate(self.record_historic_ranking_list)}

    def generate_collective_actions_transition_matrix(self):
        # Importing the required libraries and retrying the process

        # Given list of tuples (sequence of events)
        events = self.record_historic_quartile_dict

        # Generate all possible tuple states
        all_states = [(i, j, k) for i in range(1, 5) for j in range(1, 5) for k in range(1, 5)]

        # Create a dictionary to track transitions
        transitions = defaultdict(lambda: defaultdict(int))

        # Populate the transitions dictionary
        for i in range(len(events) - 1):
            current_state = events[i]
            next_state = events[i + 1]
            transitions[current_state][next_state] += 1

        # Normalize the counts to get probabilities
        for current_state, next_states in transitions.items():
            total_transitions = sum(next_states.values())
            for next_state, count in next_states.items():
                transitions[current_state][next_state] = count / total_transitions

        # Create the 64x64 transition matrix
        transition_matrix = []

        for current_state in all_states:
            row = []
            for next_state in all_states:
                row.append(transitions[current_state].get(next_state, 0))
            transition_matrix.append(row)

        self.collective_actions_transition_matrix = transition_matrix
    
    def compute_vitality_difference(self):
        round_diff = {}
        for member in self.current_members:
            current_vitality = member.vitality
            prev_vitality = self.previous_vitalities.get(member, current_vitality)  # Default to current vitality if not found
            round_diff[member] = current_vitality - prev_vitality
            self.previous_vitalities[member] = current_vitality
        self.vitality_diff[self.current_round] = round_diff

    def compute_payoff_matrix(self):
        action_combinations = [(i, j, k) for i in [0,1] for j in [0,1] for k in [0,1]]
        tuple_states = list(self.record_historic_quartile_dict.values())
        payoff_matrix = np.zeros((8, 64))
        quartile_combinations = list(itertools.product(range(1, 5), repeat=3))

        for idx_a, action_a in enumerate(action_combinations):
            for idx_t, tuple_state in enumerate(tuple_states):
                combination_index = quartile_combinations.index(tuple_state)
                total_vitality_change = 0
                count = 0

                # Assuming each member has a decision history
                for member in self.current_members:
                    try:
                        decisions = self.decision_history[member]  # Access the decision history from the dictionary in the Island class
                        for round, decision in decisions.items():
                            if decision == action_a and tuple_state == self.record_historic_quartile_dict[round]:
                                if round > 4:
                                    total_vitality_change += self.vitality_diff[round-1][member] + self.vitality_diff[round-2][member] + self.vitality_diff[round-3][member]
                                count += 1
                    except KeyError:
                        pass # for newborn babies

                if count != 0:
                    avg_vitality_change = total_vitality_change / count
                else:
                    avg_vitality_change = 0


                payoff_matrix[idx_a][combination_index] = avg_vitality_change

                self.record_payoff_matrix = payoff_matrix
    
    ############################################################################
    def save_current_island(self, path):
        current_member_df = self.current_members[0].save_to_row()
        for sur_id in range(1, self.current_member_num):
            current_member_df = pd.concat([
                current_member_df,
                self.current_members[sur_id].save_to_row()],
                axis=0
            )
        
        info_df = pd.DataFrame({
            "_create_from_file": [self._create_from_file],
            "_file_name": [self._file_name],
            "_seed": [self._random_seed],
            "init_member_num": [self.init_member_num],
            "current_member_num": [self.current_member_num],
            "current_round": [self.current_round],
            "land_shape": [f"{self.land.shape[0]} {self.land.shape[1]}"],
        })

        relationship_df = pd.DataFrame()
        for key, rela in self.relationship_dict.items():
            rela_df = pd.DataFrame(rela, index=None, columns=None)
            relationship_df = pd.concat([relationship_df, rela_df], axis=0)

        # 本轮之前保存的动作
        action_list = []
        for action_name, action_dict in self.record_action_dict.items():
            sub_action_info = [[key[0], key[1], value] for key, value in action_dict.items()]
            sub_action_df = pd.DataFrame(
                sub_action_info,
                columns=[f"{action_name}_1", f"{action_name}_2", "value"]
            )
            action_list.append(sub_action_df)
        born_df = pd.DataFrame(
            [member.id for member in self.record_born], 
            columns=["born"]
        )
        action_list.append(born_df)
        death_df = pd.DataFrame(
            [member.id for member in self.record_death], 
            columns=["death"]
        )
        action_list.append(death_df)
        action_df = pd.concat(action_list, axis=1)

        # 土地
        land_df = pd.DataFrame(
            self.land.owner_id()
        )
        
        current_member_df.to_csv(path + "members.csv")
        info_df.to_csv(path + "island_info.csv")
        relationship_df.to_csv(path + "relationships.csv")
        action_df.to_csv(path + "action.csv")
        land_df.to_csv(path + "land.csv")

    @classmethod
    def load_island(cls, path):

        current_member_df = pd.read_csv(path + "members.csv")
        info_df = pd.read_csv(path + "island_info.csv")
        relationship_df = pd.read_csv(path + "relationships.csv")
        action_df = pd.read_csv(path + "action.csv")
        land_df = pd.read_csv(path + "land.csv")

        """
        Not finished yet
        """

    def save_to_pickle(self, file_name: str) -> None:

        sys.setrecursionlimit(500000)

        file = open(file_name, 'wb') 
        pickle.dump(self, file)

    @classmethod
    def load_from_pickle(cls, file_name: str) -> "Island":
        file = open(file_name, 'rb') 
        return pickle.load(file)

    ############################################################################
    ################################## 模拟 #####################################
    @property 
    def shuffled_members(self) -> List[Member]:
        """
        打乱整个current_members列表
        """
        shuffled_members = self._backup_member_list(self.current_members)
        self._rng.shuffle(shuffled_members)

        return shuffled_members

    def declare_dead(self, member: Member):
        # 立即丢失所有土地
        for loc in member.owned_land.copy():
            self._discard_land(member, loc)
        # 清除member_2
        self.member_list_modify(drop=[member])
        # 记录死亡
        self.record_death.append(member)
        # log
        self._logger.info(f"{member.name}死了.")

    def produce(self) -> None:
        """
        生产  

        1. 根据生产力和土地，增加食物存储
        """
        for member in self.current_members:
            produced = member.produce()
            self.record_total_production[-1] += produced
            self.round_total_production += produced

    def _attack(
        self, 
        member_1: Member, 
        member_2: Member
    ) -> None:
        if 'victim' not in self.relationship_dict:
            self.relationship_dict['victim'] = np.full(
                (self.init_member_num, self.init_member_num), np.nan
            )
        # 计算攻击、偷盗值
        strength_1 = member_1.strength
        steal_1 = member_1.steal

        if steal_1 > member_2.cargo:
            steal_1 = member_2.cargo

        # 结算攻击、偷盗
        member_2.vitality -= strength_1
        member_2.cargo -= steal_1

        # 修改关系矩阵
        self.relationship_modify("victim", member_2, member_1, strength_1 + steal_1)

        # 记录动作
        self._record_actions(
            "attack",
            member_1,
            member_2,
            strength_1 + steal_1,
        )

        # 结算死亡
        if member_2.autopsy():
            # 结算死亡
            self.declare_dead(member_2)
            # member_1 立即获得扩张机会一次
            self._expand(member_1)

        # 若攻击目标的颜色和自身相同，攻击者恢复颜色（退出组织）
        if np.allclose(member_1._current_color, member_2._current_color):
            member_1._current_color = member_1._color.copy()


    def fight(
        self, 
        prob_to_fight: float = 1.0
        ):
        """
        战斗
        """
        for member in self.shuffled_members:
            if member.autopsy():
                # 如果成员在之前回合死亡，不会进行任何操作
                continue

            self._maintain_neighbor_list(member)
            
            # 从邻居中寻找目标
            target_list = self._convert_id_list_to_member_list(
                member.current_clear_list 
                + member.current_self_blocked_list 
                + member.current_neighbor_blocked_list
            )
            attack_list = self._find_targets(
                member = member,
                target_list = target_list,
                decision_name = "attack",
                prob_of_action = prob_to_fight,
                other_requirements = None,
                bilateral = False,
                land_owner_decision = "attack"
            )

            for target in attack_list:
                self._attack(member, target)

    def _offer(
        self, 
        member_1: Member, 
        member_2: Member,
        parameter_influence: bool = True
    ) -> None:
        """
        member_1 给予 member_2
        若member_1能给予的数量<1，不会给予
        """
        amount = member_1.offer

        if amount < 1:
            return 

        # 结算给予
        member_2.cargo += amount
        member_1.cargo -= amount

        # 修改关系矩阵
        self.relationship_modify("benefit", member_2, member_1, amount)

        # 记录
        if amount > 1e-15:
            self._record_actions(
                "benefit",
                member_1,
                member_2,
                amount,
                None
            )

        # 被给予者的参数受到影响
        if parameter_influence:
            member_2.parameter_absorb(
                [member_2, member_1],
                [1 - Member._PARAMETER_INFLUENCE, Member._PARAMETER_INFLUENCE],
                0
            )

        # 被给予者被染色
        member_2._current_color = member_1._current_color

    def _convert_id_list_to_member_list(self, id_list: List[int | Tuple[int,int]]) -> List[Member]:
        """
        将id_list转换为member_list
        """
        member_list = []
        for id in id_list:
            if isinstance(id, tuple):
                land_owner = self.all_members[id[0]]
                obj = self.all_members[id[1]]
                member_list.append((land_owner, obj))
            else:
                member_list.append(self.all_members[id])

        return member_list
        
    def trade(
        self,
        prob_to_trade: float = 1.0
        ):
        """
        交易与交流
        """
        for member in self.shuffled_members:            
            self._maintain_neighbor_list(member)
            
            # 从邻居中寻找目标
            trade_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(
                    member.current_clear_list 
                    + member.current_self_blocked_list 
                    + member.current_neighbor_blocked_list
                ),
                decision_name = "offer",
                prob_of_action = prob_to_trade,
                other_requirements = _requirement_for_offer,
                bilateral = False,
                land_owner_decision = "offer"
            )

            self._rng.shuffle(trade_list)
            for target in trade_list:
                self._offer(member, target, parameter_influence=True)

    def _expand(
        self,
        member: Member,
    ):
        """
        扩张
        """
        self._maintain_neighbor_list(member)
        if len(member.current_empty_loc_list) > 0:
            self._acquire_land(member, member.current_empty_loc_list[0])
            self._record_single_action("clear", member, 1.0)

    def colonize(
        self,
    ) -> None:
        """
        集体扩张
        """
        for member in self.shuffled_members:
            self._expand(member)

    def consume(
        self, 
    ):
        """
        消费

            1. 计算消耗量。消耗量会随着年龄逐步提升 
            2. 从血量中扣除消耗量，若血量小于零则记为死亡
            3. 从仓库中吃食物回满血
            4. 若有死亡案例，更新集体列表，更新编号，更新关系矩阵
        """
        for member in self.current_members:
            consumption = member.consume()

            # 记录
            self.record_total_consumption[-1] += consumption
            self.round_total_consumption += consumption

            if member.autopsy():
                self.declare_dead(member)

        for member in self.current_members:
            member.recover()

    def _offer_land(
        self, 
        member_1: Member, 
        member_2: Member,
        parameter_influence: bool = True,
        assigned_pos: float = None,
    ) -> None:
        """
        member_1 给予 member_2。  
        选出离自己最远的，离对方最近的land。  
        在提供"理想"位置时，会自动在给予者的土地中选出离assigned_pos最近的土地。
        """
        
        # 选出离自己最远的，离对方最近的land
        if member_1.land_num == 0:
            raise RuntimeError("给予土地的人应该拥有至少一块土地")
        if member_1.land_num == 0 and assigned_pos is None:
            raise RuntimeError("在没有指定位置的情况下，接受土地的人应该拥有至少一块土地")

        pos_1 = member_1.center_of_land(self.land)
        if assigned_pos is None:
            pos_2 = member_2.center_of_land(self.land)
        else:
            pos_2 = assigned_pos

        farthest_distance = -np.inf
        for land in member_1.owned_land:
            distance = self.land.distance(pos_1, land) - self.land.distance(pos_2, land)
            # distance = np.sum((pos_1 - land)**2) - np.sum((pos_2 - land)**2)
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_pos = land
        pos = farthest_pos

        # 结算给予
        self._discard_land(member_1, pos)
        self._acquire_land(member_2, pos)

        # 修改关系矩阵
        self.relationship_modify("benefit_land", member_2, member_1, 1)

        # 记录
        self._record_actions(
            "benefit_land",
            member_1,
            member_2,
            1,
            None
        )

        # 被给予者的参数受到影响
        if parameter_influence:
            member_2.parameter_absorb(
                [member_2, member_1],
                [1 - Member._PARAMETER_INFLUENCE, Member._PARAMETER_INFLUENCE],
                0
            )
            
        # 被给予者被染色
        member_2._current_color = member_1._current_color

    def land_distribute(
        self,
        prob_to_distr: float = 1.0
        ):
        """
        交易与交流
        """
        for member in self.shuffled_members:
            self._maintain_neighbor_list(member)
            
            # 从邻居中寻找目标
            distr_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(
                    member.current_clear_list 
                    + member.current_self_blocked_list 
                    + member.current_neighbor_blocked_list
                ),
                decision_name = "offer_land",
                prob_of_action = prob_to_distr,
                other_requirements = _requirement_for_offer_land,
                bilateral = False,
                land_owner_decision = "offer_land"
            )

            distr_len = len(distr_list)
            if distr_len > 0:
                target_idx = self._rng.choice(
                    range(distr_len),
                    size = 1
                )
                target = distr_list[int(target_idx)]
                self._offer_land(member, target, parameter_influence=True)

    def _bear(
        self,
        member_1: Member, 
        member_2: Member,
    ):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = Member.born(
            member_1,
            member_2,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        self.member_list_modify(
            append=[child],
            appended_rela_columns=np.zeros((self.current_member_num, 1)),
            appended_rela_rows=np.zeros((1, self.current_member_num)),
        )
        
        # 计算双亲给予孩子初始血量
        vitality_base = Member._CHILD_VITALITY
        # 双方资源总量
        member_1_total = member_1.cargo + member_1.vitality
        member_2_total = member_2.cargo + member_2.vitality
        total = member_1_total + member_2_total
        # 按比例扣除损失
        member_1_give = vitality_base * member_1_total / total
        member_2_give = vitality_base * member_2_total / total
        if member_1.cargo >= member_1_give:
            member_1.cargo -= member_1_give
        else:
            member_1.vitality -= (member_1_give - member_1.cargo)
            member_1.cargo = 0
        if member_2.cargo >= member_2_give:
            member_2.cargo -= member_2_give
        else:
            member_2.vitality -= (member_2_give - member_2.cargo)
            member_2.cargo = 0
        child.vitality = vitality_base
        # 孩子记住父母的付出
        self.relationship_modify("benefit", child, member_1, member_1_give)
        self.relationship_modify("benefit", child, member_2, member_2_give)

        # 父母无条件地offer孩子一次
        self._offer(member_1, child, parameter_influence=False)
        self._offer(member_2, child, parameter_influence=False)

        # 父母无条件给予孩子土地
        center_pos = (member_1.center_of_land(self.land) + member_2.center_of_land(self.land)) / 2
        for _ in range(Member._LAND_HERITAGE):
            self._offer_land(
                member_1, child, parameter_influence=False, 
                assigned_pos=center_pos)
            self._offer_land(
                member_2, child, parameter_influence=False, 
                assigned_pos=center_pos)

        # 孩子计算、恢复血量
        child.recover()

        self._logger.info(f"\t{member_1} 和 {member_2} 生育了 {child}")
        self._record_actions("reproduce", member_1, member_2, 1, 1)

    def reproduce(
        self, 
        prob_of_reproduce: float = 1.0
    ):
        """
        生育

            1. 择出满足年龄条件的人
            2. 随机分组，组内排序。
            3. 每组内便利，根据【生育决策】函数，判断互相好感，选择父母
            4. 判断双方是否满足生育条件（血量和仓库之和）
            5. 父母按比例扣除仓库数，总和为固定值，仓库不足时扣除血量。
            6. 产生孩子。设定孩子年龄（0），父母。孩子随机继承父母的基本属性与决策参数，添加**少许**随机浮动。孩子的初始血量为固定值（小于父母消耗值），存储……
            7. 父母无条件地offer孩子一次
            8. 若有出生案例，更新集体列表，更新编号，更新关系矩阵
            
        """

        for member in self.shuffled_members:
            self._maintain_neighbor_list(member)
            
            # 从邻居中寻找目标
            partner_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(member.current_clear_list),
                decision_name = "reproduce",
                prob_of_action = prob_of_reproduce,
                other_requirements = _requirement_for_reproduction,
                bilateral = True,
                land_owner_decision = ""
            )

            distr_len = len(partner_list)
            if distr_len > 0:
                target_idx = self._rng.choice(
                    range(distr_len),
                    size = 1
                )
                target = partner_list[int(target_idx)]
                self._bear(member, target)

    def _record_init_per_period(
        self,
    ):
        for key in self.record_action_dict.keys():
            self.record_action_dict[key] = {}
        for key in self.record_total_dict.keys():
            self.record_total_dict[key].append(0)

        self.record_total_consumption.append(0)
        self.record_total_production.append(0)

        self.record_born = []
        self.record_death = []

    def _reset_round_records(self) -> None:
        for key in self.round_action_dict.keys():
            self.round_action_dict[key] = {}
        for key in self.round_total_dict.keys():
            self.round_total_dict[key] = 0.0
        self.round_total_production = 0.0
        self.round_total_consumption = 0.0

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for a list of values."""
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        min_val = float(np.min(arr))
        if min_val < 0:
            arr = arr - min_val
        total = float(np.sum(arr))
        if total <= 0:
            return 0.0
        arr = np.sort(arr)
        n = arr.size
        cum = np.cumsum(arr)
        gini = (n + 1 - 2 * float(np.sum(cum)) / total) / n
        return float(max(0.0, min(1.0, gini)))

    def _compute_round_context(self) -> Dict[str, Any]:
        total_land = float(np.prod(self.land.shape))
        empty_land = float(np.sum(self.land.owner == None))
        land_scarcity = 1.0 - (empty_land / total_land) if total_land > 0 else 0.0

        attack_rate = self.round_total_dict.get("attack", 0.0) / max(1, self.current_member_num)
        benefit_rate = self.round_total_dict.get("benefit", 0.0) / max(1, self.current_member_num)
        benefit_land_rate = self.round_total_dict.get("benefit_land", 0.0) / max(1, self.current_member_num)

        production = self.round_total_production
        consumption = self.round_total_consumption
        resource_pressure = 0.0 if production <= 0 else (consumption - production) / production

        vitality_vals = [float(m.vitality) for m in self.current_members] if self.current_members else []
        cargo_vals = [float(m.cargo) for m in self.current_members] if self.current_members else []
        land_vals = [float(m.land_num) for m in self.current_members] if self.current_members else []

        avg_vitality = float(np.mean(vitality_vals)) if vitality_vals else 0.0
        avg_cargo = float(np.mean(cargo_vals)) if cargo_vals else 0.0
        vitality_std = float(np.std(vitality_vals)) if vitality_vals else 0.0
        cargo_std = float(np.std(cargo_vals)) if cargo_vals else 0.0
        land_std = float(np.std(land_vals)) if land_vals else 0.0
        vitality_median = float(np.median(vitality_vals)) if vitality_vals else 0.0
        cargo_median = float(np.median(cargo_vals)) if cargo_vals else 0.0
        land_median = float(np.median(land_vals)) if land_vals else 0.0

        gini_vitality = self._compute_gini(vitality_vals)
        gini_cargo = self._compute_gini(cargo_vals)
        gini_land = self._compute_gini(land_vals)
        wealth_vals = [c + l for c, l in zip(cargo_vals, land_vals)]
        gini_wealth = self._compute_gini(wealth_vals)

        action_map = {
            "attack": "attack",
            "benefit": "offer",
            "benefit_land": "offer_land",
            "reproduce": "reproduce",
            "clear": "clear",
        }
        action_counts = {name: 0.0 for name in action_map.values()}
        for action_name, decision_name in action_map.items():
            for _ in self.round_action_dict.get(action_name, {}).keys():
                action_counts[decision_name] += 1.0
        total_actions = float(sum(action_counts.values()))
        if total_actions > 0:
            action_shares = {
                name: count / total_actions for name, count in action_counts.items()
            }
            probs = [share for share in action_shares.values() if share > 0.0]
            entropy = -sum(p * math.log(p) for p in probs) / math.log(len(action_shares))
            dominant_share = max(action_shares.values()) if action_shares else 0.0
        else:
            action_shares = {name: 0.0 for name in action_counts.keys()}
            entropy = 0.0
            dominant_share = 0.0

        profile_counts = {name: 0 for name in Member._STRATEGY_PROFILES}
        for member in self.current_members:
            profile = getattr(member, "strategy_profile", None)
            if profile is None:
                continue
            if profile not in profile_counts:
                profile_counts[profile] = 0
            profile_counts[profile] += 1
        total_profiles = float(sum(profile_counts.values()))
        if total_profiles > 0:
            profile_shares = {
                name: count / total_profiles for name, count in profile_counts.items()
            }
            probs = [share for share in profile_shares.values() if share > 0.0]
            denom = math.log(len(profile_shares)) if len(profile_shares) > 1 else 0.0
            profile_entropy = -sum(p * math.log(p) for p in probs) / denom if denom > 0 else 0.0
            dominant_profile_share = max(profile_shares.values()) if profile_shares else 0.0
        else:
            profile_shares = {name: 0.0 for name in profile_counts.keys()}
            profile_entropy = 0.0
            dominant_profile_share = 0.0

        return {
            "attack_rate": attack_rate,
            "benefit_rate": benefit_rate,
            "benefit_land_rate": benefit_land_rate,
            "land_scarcity": land_scarcity,
            "resource_pressure": resource_pressure,
            "avg_vitality": avg_vitality,
            "avg_cargo": avg_cargo,
            "vitality_std": vitality_std,
            "cargo_std": cargo_std,
            "land_std": land_std,
            "vitality_median": vitality_median,
            "cargo_median": cargo_median,
            "land_median": land_median,
            "gini_vitality": gini_vitality,
            "gini_cargo": gini_cargo,
            "gini_land": gini_land,
            "gini_wealth": gini_wealth,
            "action_shares": action_shares,
            "action_entropy": entropy,
            "dominant_action_share": dominant_share,
            "profile_shares": profile_shares,
            "profile_entropy": profile_entropy,
            "dominant_profile_share": dominant_profile_share,
        }

    def _update_member_learning_and_memory(self) -> None:
        self.round_context = self._compute_round_context()
        rewards: Dict[int, float] = {}

        action_counts = {
            member.id: {
                "attack": 0.0,
                "offer": 0.0,
                "offer_land": 0.0,
                "reproduce": 0.0,
                "clear": 0.0,
            }
            for member in self.current_members
        }
        action_map = {
            "attack": "attack",
            "benefit": "offer",
            "benefit_land": "offer_land",
            "reproduce": "reproduce",
            "clear": "clear",
        }
        for action_name, decision_name in action_map.items():
            for (member_1, _), _ in self.round_action_dict.get(action_name, {}).items():
                if member_1 in action_counts:
                    action_counts[member_1][decision_name] += 1.0

        for member in self.current_members:
            prev = self._round_snapshot.get(
                member.id,
                (member.vitality, member.cargo, member.land_num)
            )
            delta_vitality = member.vitality - prev[0]
            delta_cargo = member.cargo - prev[1]
            delta_land = member.land_num - prev[2]
            member.decay_interaction_memory()
            member.update_action_memory(action_counts.get(member.id, {}))
            rewards[member.id] = member.update_round_memory(
                delta_vitality,
                delta_cargo,
                delta_land,
                self.round_context,
                action_counts=action_counts.get(member.id, {}),
            )

        # 记录交互记忆
        for (member_1, member_2), value in self.round_action_dict.get("attack", {}).items():
            actor = self.all_members[member_1]
            target = self.all_members[member_2]
            actor.record_interaction("attack_made", member_2, value)
            target.record_interaction("attack_received", member_1, value)

        for (member_1, member_2), value in self.round_action_dict.get("benefit", {}).items():
            giver = self.all_members[member_1]
            receiver = self.all_members[member_2]
            giver.record_interaction("benefit_given", member_2, value)
            receiver.record_interaction("benefit_received", member_1, value)

        for (member_1, member_2), value in self.round_action_dict.get("benefit_land", {}).items():
            giver = self.all_members[member_1]
            receiver = self.all_members[member_2]
            giver.record_interaction("land_given", member_2, value)
            receiver.record_interaction("land_received", member_1, value)

        for (member_1, member_2), value in self.round_action_dict.get("reproduce", {}).items():
            parent = self.all_members[member_1]
            partner = self.all_members[member_2]
            parent.record_interaction("benefit_given", member_2, value * 0.5)
            partner.record_interaction("benefit_given", member_1, value * 0.5)

        # 奖励驱动的参数更新
        def _apply_update(action_name: str, decision_name: str):
            for (member_1, member_2), _ in self.round_action_dict.get(action_name, {}).items():
                if member_1 not in rewards:
                    continue
                actor = self.all_members[member_1]
                try:
                    target = None if member_2 == -1 else self.all_members[member_2]
                    if target is not None and target.autopsy():
                        continue
                    inputs = actor._generate_decision_inputs(target, self)
                    input_vector = [inputs[name] for name in Member._DECISION_INPUT_NAMES]
                    actor.apply_reward_update(decision_name, input_vector, rewards[member_1])
                except Exception:
                    continue

        _apply_update("attack", "attack")
        _apply_update("benefit", "offer")
        _apply_update("benefit_land", "offer_land")
        _apply_update("reproduce", "reproduce")
        _apply_update("clear", "clear")

        # 更新下一轮快照
        self._round_snapshot = {
            member.id: (member.vitality, member.cargo, member.land_num)
            for member in self.current_members
        }

    def new_round(self, save_file: bool = True, log_status=False):
        # 行为学习与策略更新
        self._update_member_learning_and_memory()
        self._reset_round_records()

        # 输出内容
        if self.current_round % Island._RECORD_PERIOD == 0:
            # 保存
            if save_file:
                self.save_to_pickle(self._save_path + f"{self.current_round:d}.island")

            # 输出
            if log_status:
                self.log_status()

            # 初始化存储
            self._record_init_per_period()

        # 回合数+1
        self.current_round += 1

        # 每个存活成员增加一岁
        for member in self.current_members:
            member.age += 1

    def log_status(
        self,
        action = False,
        summary = True,
        members = True,
        log_instead_of_print = True,
        ):
        log_str = ""

        log_str += "#" * 21 + f" {self.current_round:d} " + "#" * 21 + "\n"

        if action:
            log_str += "=" * 21 + " 攻击 " + "=" * 21 + "\n"
            if self.record_action_dict["attack"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["attack"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

            log_str += "=" * 21 + " 给予 " + "=" * 21 + "\n"
            if self.record_action_dict["benefit"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n" 

            log_str += "=" * 20 + " 给予土地 " + "=" * 20 + "\n"
            if self.record_action_dict["benefit_land"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit_land"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

        if summary:
            log_str += "=" * 50 + "\n"
            log_str += f"本轮出生：{self.record_born} \n"
            log_str += f"本轮死亡：{self.record_death} \n"
            log_str += f"本轮总给予：{self.record_total_dict['benefit'][-1]:.1f} \n"
            log_str += f"本轮总攻击：{self.record_total_dict['attack'][-1]:.1f} \n"
            log_str += f"本轮总产量：{self.record_total_production[-1]:.1f} \n"
            log_str += f"本轮总消耗：{self.record_total_consumption[-1]:.1f} \n"
            log_str += f"本轮活跃比率：{self.record_historic_ratio_list[-1]} \n"
            log_str += f"本轮比率历史排位：{self.record_historic_ranking_list[-1]} \n"
            
        if members:
            log_str += "=" * 50 + "\n"
            log_str += "\t ID Sur_ID  姓名          年龄   血量    仓库    土地数 \n"
            for member in self.current_members:
                space_after_name = " " * (10 - len(member.name))

                mem_str = (
                    f"\t[{member.id}, {member.surviver_id}] "
                    f"{member.name}:{space_after_name}"
                    f"   {member.age}," 
                    f"   {member.vitality:.1f},"
                    f"   {member.cargo:.1f}"
                    f"   {member.land_num:d}({100*member.land_num/np.prod(self.land.shape):.1f}%)"
                    "\n"
                )

                if not log_instead_of_print:
                    mem_str = colored(
                            member._current_color,
                            mem_str
                        )
            
                log_str += mem_str

        if log_instead_of_print:
            self._logger.info(log_str)
        else:
            print(log_str)
                
    def record_statistics(self):
        """
        统计
        """
        if self.current_member_num > 0:
            self.record_historic_ratio()
            self.record_historic_ranking()
            self.calculate_histoic_quartile()
            self.generate_collective_actions_transition_matrix()
            self.generate_decision_history()
            self.compute_vitality_difference()
            self.compute_payoff_matrix()
