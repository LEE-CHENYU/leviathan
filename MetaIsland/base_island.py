import numpy as np
import pandas as pd
from MetaIsland.base_member import Member, colored
from MetaIsland.base_land import Land
from MetaIsland.settings import name_list

from utils.save import path_decorator

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import time
import logging
import pickle 
# import dill
import sys
import itertools

from collections import defaultdict

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

        # 回合数
        self.current_round = 0

        # 记录动作 （每Island._RECORD_PERIOD输出、清空一次）
        self.record_action_dict = {
            "attack": {},
            "benefit": {},
            "benefit_land": {},
        }
        self.record_born = []
        self.record_death = []

        # 记录状态 （每Island._RECORD_PERIOD向末尾增append一个0）
        self.record_total_production = [0]
        self.record_total_consumption = [0]
        self.record_total_dict = {
            "attack": [0],
            "benefit": [0],
            "benefit_land": [0],
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

        # 记录双方的动作
        try:
            record_dict[(member_1.id, member_2.id)] += value_1
        except KeyError:
            record_dict[(member_1.id, member_2.id)] = value_1
        if value_2 is not None:
            try:
                record_dict[(member_2.id, member_1.id)] += value_2
            except KeyError:
                record_dict[(member_2.id, member_1.id)] = value_2

        # 记录总动作
        if value_2 is not None:
            self.record_total_dict[record_name][-1] += value_1 + value_2
        else:
            self.record_total_dict[record_name][-1] += value_1
        
    
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
            self.record_total_production[-1] += member.produce()

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

    def _offer(
        self, 
        member_1: Member, 
        member_2: Member
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

            if member.autopsy():
                self.declare_dead(member)

        for member in self.current_members:
            member.recover()

    def _offer_land(
        self, 
        member_1: Member, 
        member_2: Member,
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
            
        # 被给予者被染色
        member_2._current_color = member_1._current_color

    def land_distribute(
        self,
        prob_to_distr: float = 1.0
        ):
        """
        交易与交流
        """

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
        self._offer(member_1, child)
        self._offer(member_2, child)

        # 父母无条件给予孩子土地
        center_pos = (member_1.center_of_land(self.land) + member_2.center_of_land(self.land)) / 2
        for _ in range(Member._LAND_HERITAGE):
            self._offer_land(
                member_1, child,
                assigned_pos=center_pos)
            self._offer_land(
                member_2, child, 
                assigned_pos=center_pos)

        # 孩子计算、恢复血量
        child.recover()

        self._logger.info(f"\t{member_1} 和 {member_2} 生育了 {child}")

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

    def new_round(self, save_file: bool = True, log_status=False):
        # 输出内容
        if self.current_round % Island._RECORD_PERIOD == 0:
            # 保存
            if save_file:
                # self.save_to_pickle(self._save_path + f"{self.current_round:d}.island")
                pass

            # 输出
            if log_status:
                self.log_status()

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