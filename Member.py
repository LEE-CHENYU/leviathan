from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import Island

def colored(rgb, text):
    r, g, b = rgb
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

class Member():
    # 上限
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 7, 8        # 生产力属性
    _PRODUCE_ELASTICITY = 0.5                          # 生产力弹性，生产力随人口增长而减少的幂的相反数
    _BASE_POPULATION = 50                              # 人口标准，在这个人口时，每轮的产量等于productivity
    _MAX_VITALITY = 100

    # 决策函数缩放参数
    _CARGO_SCALE = 0.02                                 # 在计算决策函数时，cargo的缩放量
    _RELATION_SCALES = [0.01, 0.01]                     # 决策函数在计算相互关系是的缩放量

    # 初始值
    _INIT_MIN_LAND, _INIT_MAX_LAND = 7, 8
    _INIT_MIN_VIT, _INIT_MAX_VIT = 10, 90             # 初始血量
    _INIT_MIN_CARGO, _INIT_MAX_CARGO = 0, 100         # 初始食物存储
    _INIT_MIN_AGE, _INIT_MAX_AGE = 10, 499           # 初始年龄
    _CHILD_VITALITY = 50                                # 出生时血量

    # 消耗相关计算参数
    _CONSUMPTION_BASE = 15                              # 基础消耗量
    _MAX_AGE = _INIT_MAX_AGE                            # 理论年龄最大值（消耗值等于最大生命值的年龄）
    _COMSUMPTION_CLIMBING_AGE = int(0.5 * _MAX_AGE)     # 消耗量开始显著增长的年龄
    __AGING_EXPOENT = np.log(_MAX_VITALITY - _CONSUMPTION_BASE) / (_MAX_AGE - _COMSUMPTION_CLIMBING_AGE)

    # 行动量表
    _MIN_STRENGTH, _MAX_STRENGTH = 0.1, 0.2             # 攻击力占当前血量之比
    _MIN_STEAL, _MAX_STEAL = 0.1, 0.2                   # 偷盗值占当前血量之比
    _MIN_OFFER_PERCENTAGE, _MAX_OFFER_PERCENTAGE = 0.1, 0.2                   # 给予值占当前仓库之比
    _MIN_OFFER = _MIN_PRODUCTIVITY                      # 给予最小值

    # 生育
    _MIN_REPRODUCE_AGE = int(0.18 * _MAX_AGE)           # 最小年龄
    _PARAMETER_FLUCTUATION = 0.1                       # 参数继承的浮动

    # 交易
    _PARAMETER_INFLUENCE = 0.1                       # 交易后的参数影响

    # 决策参数的名字
    _DECISION_INPUT_NAMES = [
        "self_productivity",
        "self_vitality",
        "self_cargo",
        "self_age",
        "obj_productivity",
        "obj_vitality",
        "obj_cargo",
        "obj_age",
        "victim_overlap",                   # 关系网重叠部分
        "benefit_overlap",
        # 细化关系网内积
        # "victim_passive_passive",           # victim_passive_passive代表victim记忆中，self所拥的行，乘以obj所拥的行
        # "victim_passive_active",            # 第一个p/a代表self的记忆，第二个p/a代表obj的记忆
        # "victim_active_passive",            # passive - 行，active - 列
        # "victim_active_active",
        # "benefit_passive_passive",
        # "benefit_passive_active",
        # "benefit_active_passive",
        # "benefit_active_active",
        "victim_passive",                   # self对obj的记忆
        "victim_active",                    # obj对self的记忆
        "benefit_passive",
        "benefit_active",
    ]
    _DECISION_NAMES = [
        "attack",
        "offer",
        "reproduce",
        "clear"
    ]
    _parameter_name_dict = {}               # 参数的名字
    for key in _DECISION_NAMES:
        _parameter_name_dict[key] = []
        for name in _DECISION_INPUT_NAMES:
            _parameter_name_dict[key].append(key + "_" + name)

    # 初始决策参数，列表的每行表示各个参数，列表示最小值、最大值
    # attack
    _ATTACK_PARAMETER = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 0],
        [-1, 0],
        [0, 1],
        [0, 0],
        [-1, 0],
        [-1, 0],
        [0, 1],
        [0, 0],
        [-1, 0],
        [0, 0]
    ])
    # offer
    _OFFER_PARAMETER = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [-1, 0],
        [-1, 0],
        [-1, 0],
        [0, 0],
        [0, 1],
        [0, 1],
        [-1, 0],
        [0, 0],
        [0, 1],
        [0, 0]
    ])
    # reproduce
    _REPRODUCE_PARAMETER = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 1],
        [-1, 0],
        [-1, 0],
        [0, 1],
        [0, 1]
    ])
    # clear
    _CLEAR_PARAMETER = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 1],
        [-1, 0],
        [-1, 0],
        [0, 1],
        [0, 1]
    ])

    @classmethod
    def _inherited_parameter_w_fluctuation(
        cls, 
        parameter_1: float, 
        parameter_2: float, 
        min_range: float, 
        max_range: float, 
        rng: np.random.Generator
    ):
        new_parameter = rng.uniform(min_range, max_range)
        new_parameter += (
            ((parameter_1 + parameter_2) / 2 - (min_range + max_range) / 2)
            * rng.uniform(0, cls._PARAMETER_FLUCTUATION)
        )
        
        if new_parameter > max_range:
            new_parameter = max_range
        elif new_parameter < min_range:
            new_parameter = min_range
        return new_parameter

    @classmethod
    def born(
        cls, 
        parent_1: Member, 
        parent_2: Member,
        name: str,
        id: int,
        surviver_id: int,
        rng: np.random.Generator
    ) -> Member:

        child = Member(name, id, surviver_id, rng)
        child.parent_1 = parent_1
        child.parent_2 = parent_2

        child.productivity = cls._inherited_parameter_w_fluctuation(
            parent_1.productivity, 
            parent_2.productivity, 
            cls._MIN_PRODUCTIVITY, 
            cls._MAX_PRODUCTIVITY,
            rng
        )

        child.parameter_absorb(
            [parent_1, parent_2],
            fluctuation_amplitude=cls._PARAMETER_FLUCTUATION
        )

        child.vitality = 0
        child.cargo = 0
        child.age = 0

        return child

    def __init__(
        self, 
        name: str, 
        id: int, 
        surviver_id: int,
        rng: np.random.Generator
        ) -> None:

        # 随机数生成器
        self._rng = rng

        # 姓名
        self.name = name
        self.id = id
        self.surviver_id = surviver_id      # Island.current_members中的编号

        # 亲人链表
        self.parent_1 = None
        self.parent_2 = None
        self.child = []

        # 人物颜色
        self._color = [0, 0, 0]
        while np.std(self._color) < 100:
            self._color = np.round(self._rng.uniform(0, 255, size=3)).astype(int)
        self._current_color = self._color.copy()

        # 领地
        self.owned_land = []
        self.current_clear_list = []
        self.current_self_blocked_list = []
        self.current_neighbor_blocked_list = []
        self.current_empty_loc_list = []

        # 生产相关的属性和状态
        self.land = self._rng.uniform(Member._INIT_MIN_LAND, Member._INIT_MAX_LAND)
        self.productivity = self._rng.uniform(Member._MIN_PRODUCTIVITY, Member._MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(Member._INIT_MIN_VIT, Member._INIT_MAX_VIT)
        self.cargo = self._rng.uniform(Member._INIT_MIN_CARGO, Member._INIT_MAX_CARGO)
        self.age = int(self._rng.uniform(Member._INIT_MIN_AGE, Member._INIT_MAX_AGE))

        # 决策参数
        # 攻击决策
        _attack_parameter = (
            self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
            * (Member._ATTACK_PARAMETER[:, 1] - Member._ATTACK_PARAMETER[:, 0])
            + Member._ATTACK_PARAMETER[:, 0]
        )
        # 给予决策
        _offer_parameter = (
            self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
            * (Member._OFFER_PARAMETER[:, 1] - Member._OFFER_PARAMETER[:, 0])
            + Member._OFFER_PARAMETER[:, 0]
        )
        # 生育决策
        _reproduce_parameter = (
            self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
            * (Member._REPRODUCE_PARAMETER[:, 1] - Member._REPRODUCE_PARAMETER[:, 0])
            + Member._REPRODUCE_PARAMETER[:, 0]
        )
        # 通行决策
        _clear_parameter = (
            self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
            * (Member._CLEAR_PARAMETER[:, 1] - Member._CLEAR_PARAMETER[:, 0])
            + Member._CLEAR_PARAMETER[:, 0]
        )
        self.parameter_dict = {
            "attack": _attack_parameter,
            "offer": _offer_parameter,
            "reproduce": _reproduce_parameter,
            "clear": _clear_parameter
        }
        assert list(self.parameter_dict.keys()) == Member._DECISION_NAMES

    def __str__(self):
        """重载print函数表示"""
        return colored(self._current_color, f"{self.name}({self.id})")

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.__str__()
    
    @property
    def strength(self) -> float:
        """战斗力：每次攻击造成的伤害"""
        return (
            self._rng.uniform(Member._MIN_STRENGTH, Member._MAX_STRENGTH)
            * self.vitality
        )
    
    @property
    def steal(self) -> float:
        """每次偷盗的收获"""
        return (
            self._rng.uniform(Member._MIN_STEAL, Member._MAX_STEAL) 
            * self.vitality
        )

    @property
    def offer(self) -> float:
        """每次给予的数额"""
        return (
            self._rng.uniform(Member._MIN_OFFER_PERCENTAGE, Member._MAX_OFFER_PERCENTAGE) 
            * self.cargo
        )

    @property
    def consumption(self) -> float:
        """每轮消耗量"""
        amount = (Member._CONSUMPTION_BASE 
            + np.exp(Member.__AGING_EXPOENT * (self.age - Member._COMSUMPTION_CLIMBING_AGE))
        )
        return amount 

    def autopsy(self) -> bool:
        """验尸，在Member类中结算死亡，返回是否死亡"""
        if self.vitality <= 0:
            self.vitality = 0
            return True
        return False

    @property
    def is_qualified_to_reproduce(self):
        return self.age >= Member._MIN_REPRODUCE_AGE

    #########################################################################
    ################################## 动作 ################################## 

    def _generate_decision_inputs(
        self, 
        object: Member, 
        island: Island.Island
        ) -> Dict:

        assert self is not object, "决策函数中主体和对象不能相同"

        len_input = len(Member._DECISION_INPUT_NAMES)
        input_dict = dict(zip(Member._DECISION_INPUT_NAMES, np.zeros(len_input)))

        input_dict["self_productivity"] = self.productivity / Member._MAX_PRODUCTIVITY
        input_dict["self_vitality"] = self.vitality / Member._MAX_VITALITY
        input_dict["self_cargo"] = np.tanh(self.cargo * Member._CARGO_SCALE)
        input_dict["self_age"] = self.age / Member._MAX_AGE
        input_dict["obj_productivity"] = object.productivity / Member._MAX_PRODUCTIVITY
        input_dict["obj_vitality"] = object.vitality / Member._MAX_VITALITY
        input_dict["obj_cargo"] = np.tanh(object.cargo * Member._CARGO_SCALE)
        input_dict["obj_age"] = object.age / Member._MAX_AGE

        input_dict["victim_overlap"], input_dict["benefit_overlap"] = island._overlap_of_relations(self, object)
        
        input_dict["victim_passive"], input_dict["victim_active"], input_dict["benefit_passive"], input_dict["benefit_active"] = island._relations_w_normalize(self, object)

        return input_dict

    def decision(
        self, 
        parameter_name: str, 
        object: Member,
        island: Island.Island
    ) -> bool:

        return (
            np.sum(self.parameter_dict[parameter_name] 
            * list(self._generate_decision_inputs(object, island).values()))
        ) > 1

    def parameter_absorb(
        self,
        contributor_list: List[Member] = [],
        weight_list: List[float] = [],
        fluctuation_amplitude = 0,
    ) -> None:
        """产生多个人的决策参数加权平均值"""
        # 加权平均
        new_dict = {key: contributor_list[0].parameter_dict[key].copy() * weight_list[0] for key in contributor_list[0].parameter_dict.keys()}

        for idx in range(1, len(contributor_list)):
            contributor = contributor_list[idx]
            for key in new_dict.keys():
                new_dict[key] += contributor.parameter_dict[key] * weight_list[idx]

        # 浮动
        if fluctuation_amplitude > 0:
            for key in new_dict.keys():
                new_dict[key] += self._rng.uniform(
                    -fluctuation_amplitude, 
                    fluctuation_amplitude,
                    size = new_dict[key].size
                )
        self.parameter_dict = new_dict

    def produce(self, population) -> float:
        """生产，将收获装入cargo"""
        productivity = self.productivity + self.land
        self.cargo += productivity
        return productivity

    def consume(self) -> float:
        """消耗vitality"""
        consumption = self.consumption
        if self.vitality > consumption:
            self.vitality -= self.consumption
        else:
            consumption = self.vitality
            self.vitality = 0
        return consumption

    def recover(self) -> None:
        """使用cargo恢复vitality"""
        amount = np.min([self.cargo, Member._MAX_VITALITY - self.vitality])
        self.vitality += amount
        self.cargo -= amount

# ================================== 保存 =======================================
    def save_to_row(self):

        info_df = pd.DataFrame({
            "name": [self.name],
            "surviver_id": [self.surviver_id],
            "productivity": [self.productivity],
            "vitality": [self.vitality],
            "cargo": [self.cargo],
            "age": [self.age],
            "color_R": [self._color[0]],
            "color_G": [self._color[1]],
            "color_B": [self._color[2]],
            "current_R": [self._current_color[0]],
            "current_G": [self._current_color[1]],
            "current_B": [self._current_color[2]],
        }, index=[self.id])
        if self.parent_1 is not None:
            info_df["parent_1"] = [self.parent_1.id]
        if self.parent_2 is not None:
            info_df["parent_2"] = [self.parent_2.id]

        # 存储决策参数
        for key, paras in self.parameter_dict.items():
            parameters = pd.DataFrame(
                dict(zip(
                    Member._parameter_name_dict[key], 
                    paras.reshape(-1, 1)
                )),
                index=[self.id]
            )
            info_df = pd.concat([info_df, parameters], axis=1)

        return info_df


