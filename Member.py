from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import Island

class Member():
    # 上限
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 10, 20     # 生产力属性
    _MAX_VITALITY = 100

    # 决策函数缩放参数
    _CARGO_SCALE = 0.02                                 # 在计算决策函数时，cargo的缩放量
    _RELATION_SCALES = [0.01, 0.01]                     # 决策函数在计算相互关系是的缩放量

    # 初始值
    _INIT_MIN_VIT, _INIT_MAX_VIT = 10, 90             # 初始血量
    _INIT_MIN_CARGO, _INIT_MAX_CARGO = 0, 100         # 初始食物存储
    _INIT_MIN_AGE, _INIT_MAX_AGE = 10, 1000           # 初始年龄

    # 消耗相关计算参数
    _CONSUMPTION_BASE = 10                              # 基础消耗量
    _COMSUMPTION_CLIMBING_AGE = 800                     # 消耗量开始显著增长的年龄
    _MAX_AGE = 999                                     # 理论年龄最大值（消耗值等于最大生命值的年龄）
    __AGING_EXPOENT = np.log(_MAX_VITALITY - _CONSUMPTION_BASE) / (_MAX_AGE - _COMSUMPTION_CLIMBING_AGE)

    # 行动量表
    _MIN_STRENGTH, _MAX_STRENGTH = 0.1, 0.2             # 攻击力占当前血量之比
    _MIN_STEAL, _MAX_STEAL = 0.1, 0.2                   # 偷盗值占当前血量之比
    _MIN_OFFER, _MAX_OFFER = 0.1, 0.2                   # 给予值占当前仓库之比

    # 生育
    _MIN_REPRODUCE_AGE = int(0.18 * _MAX_AGE)           # 最小年龄
    _PARAMETER_FLUCTUATION = 0.05                       # 参数继承的浮动

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

    @classmethod
    def _inherited_parameter_w_fluctuation(
        cls, 
        parameter_1: float, 
        parameter_2: float, 
        min_range: float, 
        max_range: float, 
        rng: np.random.Generator
    ):
        new_parameter = (parameter_1 + parameter_2) / 2
        new_parameter *= rng.uniform(1 - cls._PARAMETER_FLUCTUATION, 1 + cls._PARAMETER_FLUCTUATION)
        
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

        # 生产相关的属性和状态
        self.productivity = self._rng.uniform(Member._MIN_PRODUCTIVITY, Member._MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(Member._INIT_MIN_VIT, Member._INIT_MAX_VIT)
        self.cargo = self._rng.uniform(Member._INIT_MIN_CARGO, Member._INIT_MAX_CARGO)
        self.age = int(self._rng.uniform(Member._INIT_MIN_AGE, Member._INIT_MAX_AGE))

        # 决策参数
        # 攻击决策
        _attack_parameter = self._rng.uniform(-1, 1, size=len(Member._DECISION_INPUT_NAMES))
        # 给予决策
        _offer_parameter = self._rng.uniform(-1, 1, size=len(Member._DECISION_INPUT_NAMES))
        # 生育决策
        _reproduce_parameter = self._rng.uniform(-1, 1, size=len(Member._DECISION_INPUT_NAMES))
        self.parameter_dict = {
            "attack": _attack_parameter,
            "offer": _offer_parameter,
            "reproduce": _reproduce_parameter
        }

    def __str__(self):
        """重载print函数表示"""
        return f"{self.name}({self.id})"

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
            self._rng.uniform(Member._MIN_OFFER, Member._MAX_OFFER) 
            * self.cargo
        )
        
    @property
    def consumption(self) -> float:
        """每轮消耗量"""
        return (Member._CONSUMPTION_BASE 
            + np.exp(Member.__AGING_EXPOENT * (self.age - Member._COMSUMPTION_CLIMBING_AGE))
        )

    def autopsy(self) -> bool:
        """验尸，在Member类中结算死亡，返回是否死亡"""
        if self.vitality <= 0:
            self.vitality = 0
            print(self, "死了")
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
    ) -> float:
        return np.sum(self.parameter_dict[parameter_name] * list(self._generate_decision_inputs(object, island).values()))

    def parameter_absorb(
        self,
        contributor_list: List[Member] = [],
        fluctuation_amplitude = 0,
    ) -> None:
        """产生多个人的决策参数平均值"""
        new_dict = {key: contributor_list[0].parameter_dict[key].copy() for key in contributor_list[0].parameter_dict.keys()}
        for contributor in contributor_list[1:]:
            for key in new_dict.keys():
                new_dict[key] += contributor.parameter_dict[key]
        for key in new_dict.keys():
            new_dict[key] /= len(contributor_list)
            if fluctuation_amplitude > 0:
                new_dict[key] *= self._rng.uniform(1 - fluctuation_amplitude, 1 + fluctuation_amplitude)
        self.parameter_dict = new_dict

    def produce(self) -> None:
        """生产，将收获装入cargo"""
        self.cargo += self.productivity

    def consume(self) -> bool:
        """消耗vitality"""
        self.vitality -= self.productivity
        if self.die():
            self.vitality = 0

    def recover(self) -> None:
        """使用cargo恢复vitality"""
        amount = np.min([self.cargo, Member._MAX_VITALITY - self.vitality])
        self.vitality += amount
        self.cargo -= amount
