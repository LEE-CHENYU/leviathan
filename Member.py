import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from __future__ import annotations

from Island import Island

class Member():

    # 初始值
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 10, 20     # 生产力属性
    _INIT_MIN_VIT, _INIT_MAX_VIT = 10, 90             # 初始血量
    _INIT_MIN_CARGO, _INIT_MAX_CARGO = 0, 100         # 初始食物存储
    _INIT_MIN_AGE, _INIT_MAX_AGE = 10, 1000           # 初始年龄
    # CONSUMPTION 

    # 上限
    _MAX_VITALITY = 100
    _CARGO_SCALE = 0.02                                 # 在计算决策函数时，cargo的缩放量
    _RELATION_SCALES = [0.01, 0.01]                     # 决策函数在计算相互关系是的缩放量


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
    def born(
        cls, 
        parent_1: Member, 
        parent_2: Member
        ) -> Member:
        pass

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
        # self.fight_parameter = np.
        # 给予决策
        # 生育决策

    def __str__(self):
        """重载print函数表示"""
        return f"{self.name}({self.id})"

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.__str__()
    
    @property
    def strength(self) -> float:
        """
        战斗力：每次攻击造成的伤害
        """
        pass

    @property
    def consumption(self) -> float:
        """
        每轮消耗量
        """
        pass


    #########################################################################
    ################################## 动作 ################################## 
    def _generate_decision_inputs(
        self, 
        object: Member, 
        island: Island
        ) -> Dict:

        assert self is not object, "决策函数中主体和对象不能相同"

        len_input = len(Member._DECISION_INPUT_NAMES)
        input_dict = dict(zip(Member._DECISION_INPUT_NAMES, np.zeros(len_input)))

        input_dict["self_productivity"] = self.productivity / Member._MAX_PRODUCTIVITY
        input_dict["self_vitality"] = self.vitality / Member._MAX_VITALITY
        input_dict["self_cargo"] = np.tanh(self.cargo * Member._CARGO_SCALE)
        input_dict["self_age"] = self.age
        input_dict["obj_productivity"] = object.productivity / Member._MAX_PRODUCTIVITY
        input_dict["obj_vitality"] = object.vitality / Member._MAX_VITALITY
        input_dict["obj_cargo"] = np.tanh(object.cargo * Member._CARGO_SCALE)
        input_dict["obj_age"] = object.age

        input_dict["victim_overlap"], input_dict["benefit_overlap"] = island._overlap_of_relations(self, object)
        
        input_dict["victim_passive"], input_dict["victim_active"], input_dict["benefit_passive"], input_dict["benefit_active"] = island._relations_w_normalize(self, object)

        return input_dict

    def _decision(
        self, 
        parameters: np.ndarray, 
        inputs: np.ndarray
    ) -> float:
        return np.sum(parameters * inputs)

    def produce(self) -> None:
        self.cargo += self.productivity

    

