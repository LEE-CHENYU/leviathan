from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from collections import deque, defaultdict

from Leviathan.prompt import decision_using_gemini, decision_using_gpt35
import Leviathan.Island as Island
import Leviathan.Land as Land

def colored(rgb, text):
    r, g, b = rgb
    # return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
    return text

class Member():
    # 生产力
    # prod * (land / standard)**0.5
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 15, 20       # 生产力属性
    _PRODUCE_ELASTICITY = 0.5                           # 生产力弹性，生产力随土地增长而增加的幂
    _STD_LAND = 4                                       # 土地标准，在这个土地时，每轮的产量等于productivity
    _MAX_VITALITY = 100

    # 决策
    _CARGO_SCALE = 0.02                                 # 在计算决策函数时，cargo的缩放量
    _RELATION_SCALES = [0.01, 0.01, 0.25]                  # 决策函数在计算相互关系时的缩放量
    _MAX_NEIGHBOR = 4                                   # 邻居数量最大值
    _DECISION_BACKEND = "gpt3.5"                        # 决策函数的后端(inner product, geminim, gpt3.5)

    # 行为记忆与策略调整
    _MEMORY_WINDOW = 8
    _INTERACTION_DECAY = 0.9
    _LEARNING_RATE = 0.05
    _REWARD_SCALE = 50.0
    _REWARD_WEIGHTS = (1.0, 0.35, 2.0)  # vitality, cargo, land
    _DECISION_SCALE_RANGE = (0.6, 1.4)
    _TARGET_SCALE_RANGE = (0.7, 1.3)
    _RELATION_SCORE_SCALE = 50.0
    _ACTION_MEMORY_DECAY = 0.85
    _ACTION_DIVERSITY_WEIGHT = 0.12
    _ACTION_DIVERSITY_CLAMP = (0.9, 1.1)
    _ACTION_EFFICACY_DECAY = 0.85
    _ACTION_EFFICACY_WEIGHT = 0.12
    _ACTION_EFFICACY_CLAMP = (0.85, 1.15)
    _POPULATION_DIVERSITY_WEIGHT = 0.08
    _POPULATION_DIVERSITY_CLAMP = (0.9, 1.1)
    _STRATEGY_PROFILES = {
        "aggressive": {"attack": 1.2, "offer": 0.85, "reproduce": 0.9, "clear": 1.1, "offer_land": 0.85},
        "cooperative": {"attack": 0.85, "offer": 1.2, "reproduce": 1.05, "clear": 1.0, "offer_land": 1.15},
        "expansionist": {"attack": 1.0, "offer": 0.9, "reproduce": 0.9, "clear": 1.25, "offer_land": 0.85},
        "builder": {"attack": 0.9, "offer": 1.1, "reproduce": 1.2, "clear": 1.1, "offer_land": 1.0},
    }
    _PROFILE_SCORE_DECAY = 0.9
    _PROFILE_REWARD_WEIGHT = 0.8
    _PROFILE_SWITCH_RATE = 0.12
    _PROFILE_SWITCH_RATE_CAP = 0.3
    _PROFILE_MIN_TENURE = 3
    _PROFILE_TEMPERATURE = 0.7
    _PROFILE_DIVERSITY_WEIGHT = 0.35
    _PROFILE_DOMINANCE_SWITCH_BOOST = 0.4
    _PROFILE_RARITY_SWITCH_REDUCTION = 0.25
    _PROFILE_NEGATIVE_THRESHOLD = -0.05
    _PROFILE_POSITIVE_THRESHOLD = 0.08

    # 初始值
    _INIT_MIN_VIT, _INIT_MAX_VIT = 10, 90             # 初始血量
    _INIT_MIN_CARGO, _INIT_MAX_CARGO = 0, 100         # 初始食物存储
    _INIT_MIN_AGE, _INIT_MAX_AGE = 10, 99           # 初始年龄
    _CHILD_VITALITY = 50                                # 出生时血量

    # 消耗相关计算参数
    _CONSUMPTION_BASE = 15                              # 基础消耗量
    _MAX_AGE = _INIT_MAX_AGE                            # 理论年龄最大值（消耗值等于最大生命值的年龄）
    _COMSUMPTION_CLIMBING_AGE = int(0.5 * _MAX_AGE)     # 消耗量开始显著增长的年龄
    __AGING_EXPOENT = np.log(_MAX_VITALITY - _CONSUMPTION_BASE) / (_MAX_AGE - _COMSUMPTION_CLIMBING_AGE)

    # 行动量表
    _MIN_STRENGTH, _MAX_STRENGTH = 0.1, 0.3             # 攻击力占当前血量之比
    _MIN_STEAL, _MAX_STEAL = 0.1, 0.3                   # 偷盗值占当前血量之比
    _MIN_OFFER_PERCENTAGE, _MAX_OFFER_PERCENTAGE = 0.1, 0.3                   # 给予值占当前仓库之比
    _MIN_OFFER = 5                                      # 给予最小值

    # 生育
    _MIN_REPRODUCE_AGE = int(0.18 * _MAX_AGE)           # 最小年龄
    _PARAMETER_FLUCTUATION = 0.4                       # 参数继承的浮动
    _LAND_HERITAGE = np.ceil(_STD_LAND / 2).astype(int) # 生育给予的土地数量

    # 交易
    _PARAMETER_INFLUENCE = 0.01                       # 交易后的参数影响

    # 决策参数的名字
    _DECISION_INPUT_NAMES = [
        "self_productivity",
        "self_vitality",
        "self_cargo",
        "self_age",
        "self_neighbor",

        "obj_productivity",
        "obj_vitality",
        "obj_cargo",
        "obj_age",
        "obj_neighbor",

        "victim_overlap",                   # 关系网重叠部分
        "benefit_overlap",
        "benefit_land_overlap",

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
        "benefit_land_passive",
        "benefit_land_active",
        "recent_help_received",
        "recent_harm_received",
    ]
    _DECISION_NAMES = [
        "attack",
        "offer",
        "reproduce",
        "clear",
        "offer_land",
    ]
    _parameter_name_dict = {}               # 参数的名字
    for key in _DECISION_NAMES:
        _parameter_name_dict[key] = []
        for name in _DECISION_INPUT_NAMES:
            _parameter_name_dict[key].append(key + "_" + name)

    # 初始决策参数，列表的每行表示各个参数，列表示最小值、最大值
    _INITIAL_PRAMETER = {}
    _INITIAL_PRAMETER["attack"] = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [-1, 1],

        [0, 0],
        [-1, 0],
        [0, 1],
        [0, 0],
        [-1, 0],

        [-1, 0],
        [-1, 1],
        [-1, 1],

        [0, 1],
        [0, 0],
        [-1, 0],
        [0, 0],
        [-1, 0],
        [0, 0],
        [-1, 0],
        [0, 1],
    ])
    _INITIAL_PRAMETER["offer"] = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],

        [-1, 0],
        [-1, 0],
        [-1, 0],
        [0, 0],
        [-1, 0],

        [0, 1],
        [0, 1],
        [0, 1],

        [-1, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1],
        [-1, 0],
    ])
    _INITIAL_PRAMETER["reproduce"] = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],

        [0, 1], 
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],

        [0, 1],
        [0, 1],
        [0, 1],

        [-1, 0],
        [-1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [-1, 0],
    ])
    _INITIAL_PRAMETER["clear"] = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],

        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],

        [0, 1],
        [0, 1],
        [0, 1],

        [-1, 0],
        [-1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [-1, 0],
    ])
    _INITIAL_PRAMETER["offer_land"] = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [0, 1],

        [-1, 0],
        [-1, 0],
        [-1, 0],
        [0, 0],
        [-1, 0],

        [0, 1],
        [0, 1],
        [0, 1],

        [-1, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1],
        [-1, 0],
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
        # child.parent_1 = parent_1
        # child.parent_2 = parent_2
        # parent_1.children.append(child)
        # parent_2.children.append(child)

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
        # self.parent_1 = None
        # self.parent_2 = None
        # self.children = []

        # 人物颜色
        self._color = [0, 0, 0]
        while np.std(self._color) < 100:
            self._color = np.round(self._rng.uniform(0, 255, size=3)).astype(int)
        self._current_color = self._color.copy()

        # 领地
        self.owned_land: List[Tuple[int, int]] = []
        self.land_num = 0

        self.current_clear_list = []
        self.current_self_blocked_list = []
        self.current_neighbor_blocked_list = []
        self.current_empty_loc_list = []

        # 生产相关的属性和状态
        self.productivity = self._rng.uniform(Member._MIN_PRODUCTIVITY, Member._MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(Member._INIT_MIN_VIT, Member._INIT_MAX_VIT)
        self.cargo = self._rng.uniform(Member._INIT_MIN_CARGO, Member._INIT_MAX_CARGO)
        self.age = int(self._rng.uniform(Member._INIT_MIN_AGE, Member._INIT_MAX_AGE))

        # 随机初始化决策参数
        self.parameter_dict = {}
        for des_name in Member._DECISION_NAMES:
            para_range = Member._INITIAL_PRAMETER[des_name]
            self.parameter_dict[des_name] = (
                self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
                * (para_range[:, 1] - para_range[:, 0])
                + para_range[:, 0]
            )

        # 行为策略与记忆
        self.strategy_profile = self._rng.choice(list(Member._STRATEGY_PROFILES.keys()))
        self.strategy_state = {
            "decision_scale": {},
            "adaptive_scale": {name: 1.0 for name in Member._DECISION_NAMES},
            "recent_rewards": deque(maxlen=Member._MEMORY_WINDOW),
            "recent_vitality": deque(maxlen=Member._MEMORY_WINDOW),
            "recent_cargo": deque(maxlen=Member._MEMORY_WINDOW),
            "recent_land": deque(maxlen=Member._MEMORY_WINDOW),
            "reward_baseline": 0.0,
            "profile_scale_applied": False,
            "profile_scores": {name: 0.0 for name in Member._STRATEGY_PROFILES},
            "profile_tenure": 0,
            "action_memory": {
                "attack": 0.0,
                "offer": 0.0,
                "offer_land": 0.0,
                "reproduce": 0.0,
                "clear": 0.0,
            },
            "action_efficacy": {name: 0.0 for name in Member._DECISION_NAMES},
        }
        self.interaction_memory = {
            "attack_received": defaultdict(float),
            "attack_made": defaultdict(float),
            "benefit_received": defaultdict(float),
            "benefit_given": defaultdict(float),
            "land_received": defaultdict(float),
            "land_given": defaultdict(float),
        }
        self._apply_profile_scaling()

    def __str__(self):
        """重载print函数表示"""
        return colored(self._current_color, f"{self.name}({self.id})")

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.__str__()

    # def __getstate__(self):
    #     # print(f"Pickling {self.__class__.__name__} object with ID: {self.__dict__}")
    #     state = self.__dict__.copy()
    #     # Add any custom pickling logic here
    #     return state

    # ---------------------------- 策略与记忆 ---------------------------- #
    def _clip_parameters(self, decision_name: str, params: np.ndarray) -> np.ndarray:
        ranges = Member._INITIAL_PRAMETER[decision_name]
        return np.minimum(np.maximum(params, ranges[:, 0]), ranges[:, 1])

    def _ensure_parameter_length(self, decision_name: str) -> np.ndarray:
        params = self.parameter_dict[decision_name]
        target_len = len(Member._DECISION_INPUT_NAMES)
        current_len = params.shape[0]
        if current_len == target_len:
            return params
        if current_len < target_len:
            pad = np.zeros(target_len - current_len, dtype=float)
            params = np.concatenate([params, pad])
        else:
            params = params[:target_len]
        params = self._clip_parameters(decision_name, params)
        self.parameter_dict[decision_name] = params
        return params

    def _apply_profile_scaling(self) -> None:
        profile = Member._STRATEGY_PROFILES.get(self.strategy_profile)
        if not profile:
            profile = {name: 1.0 for name in Member._DECISION_NAMES}
        self.strategy_state["decision_scale"] = profile.copy()
        self.strategy_state["profile_scale_applied"] = False

    def decay_interaction_memory(self, decay: float = None) -> None:
        if decay is None:
            decay = Member._INTERACTION_DECAY
        for memory in self.interaction_memory.values():
            for other_id in list(memory.keys()):
                memory[other_id] *= decay
                if abs(memory[other_id]) < 1e-6:
                    del memory[other_id]

    def record_interaction(self, kind: str, other_id: int, value: float) -> None:
        if kind in self.interaction_memory:
            self.interaction_memory[kind][other_id] += value

    def update_round_memory(
        self,
        delta_vitality: float,
        delta_cargo: float,
        delta_land: float,
        island_context: Optional[Dict[str, float]] = None,
        action_counts: Optional[Dict[str, float]] = None,
    ) -> float:
        recent_rewards = self.strategy_state["recent_rewards"]
        baseline = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else 0.0
        self.strategy_state["reward_baseline"] = baseline
        reward = (
            delta_vitality * Member._REWARD_WEIGHTS[0]
            + delta_cargo * Member._REWARD_WEIGHTS[1]
            + delta_land * Member._REWARD_WEIGHTS[2]
        )
        scaled_reward = float(np.tanh(reward / Member._REWARD_SCALE))
        self.strategy_state["recent_rewards"].append(scaled_reward)
        self.strategy_state["recent_vitality"].append(delta_vitality)
        self.strategy_state["recent_cargo"].append(delta_cargo)
        self.strategy_state["recent_land"].append(delta_land)
        if action_counts is not None:
            self.update_action_efficacy(action_counts, scaled_reward)
        self._update_adaptive_scales(island_context)
        self.update_profile_preferences(scaled_reward, island_context)
        return scaled_reward

    def update_action_memory(self, action_counts: Dict[str, float]) -> None:
        action_memory = self.strategy_state.get("action_memory")
        if not action_memory:
            return
        for key in action_memory.keys():
            action_memory[key] *= Member._ACTION_MEMORY_DECAY
        for action, count in action_counts.items():
            if action in action_memory:
                action_memory[action] += float(count)

    def update_action_efficacy(self, action_counts: Dict[str, float], reward: float) -> None:
        efficacy = self.strategy_state.get("action_efficacy")
        if not efficacy:
            return
        for key in efficacy.keys():
            efficacy[key] *= Member._ACTION_EFFICACY_DECAY
        for action, count in action_counts.items():
            if action not in efficacy:
                continue
            participation = min(1.0, float(count))
            efficacy[action] += (1.0 - Member._ACTION_EFFICACY_DECAY) * reward * participation

    def _update_adaptive_scales(self, island_context: Optional[Dict[str, float]] = None) -> None:
        scales = {name: 1.0 for name in Member._DECISION_NAMES}

        # 资源压力：低血量/低仓库更偏向进攻与通行
        if self.vitality < 0.4 * Member._MAX_VITALITY or self.cargo < Member._MIN_OFFER:
            scales["attack"] *= 1.15
            scales["clear"] *= 1.1
            scales["offer"] *= 0.9
            scales["offer_land"] *= 0.9

        # 土地短缺：优先获取通行权与扩张，降低送地
        land_goal = Member._LAND_HERITAGE + 1
        if self.land_num < land_goal:
            scales["clear"] *= 1.2
            scales["offer_land"] *= 0.85
            scales["reproduce"] *= 0.9
        else:
            scales["offer_land"] *= 1.05
            if self.is_qualified_to_reproduce and self.cargo > Member._MIN_OFFER:
                scales["reproduce"] *= 1.1

        # 近期收益趋势
        if len(self.strategy_state["recent_rewards"]) > 0:
            avg_reward = float(np.mean(self.strategy_state["recent_rewards"]))
            if avg_reward < -0.05:
                scales["attack"] *= 1.1
                scales["clear"] *= 1.05
                scales["offer"] *= 0.9
            elif avg_reward > 0.05:
                scales["offer"] *= 1.05
                scales["reproduce"] *= 1.05

        # 环境上下文
        if island_context:
            if island_context.get("attack_rate", 0) > 0.5:
                scales["offer"] *= 1.05
                scales["attack"] *= 1.05
            if island_context.get("land_scarcity", 0) > 0.7:
                scales["clear"] *= 1.1
                scales["offer_land"] *= 0.9
            resource_pressure = island_context.get("resource_pressure", 0.0)
            if resource_pressure > 0.2:
                scales["offer"] *= 1.05
                scales["clear"] *= 1.05
                scales["attack"] *= 0.95
                scales["offer_land"] *= 0.95
            elif resource_pressure < -0.2:
                scales["reproduce"] *= 1.05
                scales["offer_land"] *= 1.05

        action_memory = self.strategy_state.get("action_memory")
        if action_memory:
            total = float(sum(action_memory.values()))
            if total > 0:
                expected = 1.0 / len(action_memory)
                for action_name, count in action_memory.items():
                    if action_name == "reproduce" and not self.is_qualified_to_reproduce:
                        continue
                    if action_name == "offer_land" and self.land_num <= 0:
                        continue
                    share = count / total
                    diversity_bias = (expected - share) / max(expected, 1e-6)
                    adjust = 1.0 + Member._ACTION_DIVERSITY_WEIGHT * diversity_bias
                    adjust = float(np.clip(adjust, *Member._ACTION_DIVERSITY_CLAMP))
                    scales[action_name] *= adjust

        if island_context:
            population_shares = island_context.get("action_shares")
            population_entropy = island_context.get("action_entropy")
            if isinstance(population_shares, dict) and population_shares:
                expected = 1.0 / len(population_shares)
                diversity_pressure = (
                    1.0 - float(population_entropy)
                    if population_entropy is not None
                    else 1.0
                )
                diversity_weight = Member._POPULATION_DIVERSITY_WEIGHT * float(
                    np.clip(diversity_pressure, 0.0, 1.0)
                )
                if diversity_weight > 0:
                    for action_name, share in population_shares.items():
                        if action_name == "reproduce" and not self.is_qualified_to_reproduce:
                            continue
                        if action_name == "offer_land" and self.land_num <= 0:
                            continue
                        profile_scale = self.strategy_state.get("decision_scale", {}).get(
                            action_name, 1.0
                        )
                        profile_weight = 1.0 / max(0.5, float(profile_scale))
                        diversity_bias = (expected - float(share)) / max(expected, 1e-6)
                        adjust = 1.0 + diversity_weight * diversity_bias * profile_weight
                        adjust = float(np.clip(adjust, *Member._POPULATION_DIVERSITY_CLAMP))
                        scales[action_name] *= adjust

        action_efficacy = self.strategy_state.get("action_efficacy")
        if action_efficacy:
            for action_name, value in action_efficacy.items():
                if action_name == "reproduce" and not self.is_qualified_to_reproduce:
                    continue
                if action_name == "offer_land" and self.land_num <= 0:
                    continue
                adjust = 1.0 + Member._ACTION_EFFICACY_WEIGHT * float(value)
                adjust = float(np.clip(adjust, *Member._ACTION_EFFICACY_CLAMP))
                scales[action_name] *= adjust

        # clamp
        for key, value in scales.items():
            scales[key] = float(np.clip(value, *Member._DECISION_SCALE_RANGE))
        self.strategy_state["adaptive_scale"] = scales

    def _ensure_profile_state(self) -> None:
        state = self.strategy_state
        if not state.get("profile_scores"):
            state["profile_scores"] = {name: 0.0 for name in Member._STRATEGY_PROFILES}
        if "profile_tenure" not in state:
            state["profile_tenure"] = 0

    def update_profile_preferences(
        self,
        reward: float,
        island_context: Optional[Dict[str, float]] = None,
    ) -> None:
        self._ensure_profile_state()
        state = self.strategy_state
        scores = state["profile_scores"]

        for key in list(scores.keys()):
            scores[key] *= Member._PROFILE_SCORE_DECAY
        scores[self.strategy_profile] = scores.get(self.strategy_profile, 0.0) + (
            Member._PROFILE_REWARD_WEIGHT * reward
        )

        state["profile_tenure"] = int(state.get("profile_tenure", 0)) + 1
        if state["profile_tenure"] < Member._PROFILE_MIN_TENURE:
            return

        switch_prob = Member._PROFILE_SWITCH_RATE
        if reward < Member._PROFILE_NEGATIVE_THRESHOLD:
            switch_prob *= 1.5
        elif reward > Member._PROFILE_POSITIVE_THRESHOLD:
            switch_prob *= 0.6

        if island_context:
            profile_shares = island_context.get("profile_shares")
            if isinstance(profile_shares, dict) and profile_shares:
                expected = 1.0 / len(profile_shares)
                current_share = float(profile_shares.get(self.strategy_profile, 0.0))
                if current_share > expected:
                    dominance = (current_share - expected) / max(expected, 1e-6)
                    switch_prob *= 1.0 + Member._PROFILE_DOMINANCE_SWITCH_BOOST * dominance
                elif current_share < expected:
                    rarity = (expected - current_share) / max(expected, 1e-6)
                    reduction = Member._PROFILE_RARITY_SWITCH_REDUCTION * rarity
                    switch_prob *= max(0.0, 1.0 - reduction)

        switch_prob = float(np.clip(switch_prob, 0.0, Member._PROFILE_SWITCH_RATE_CAP))

        if self._rng.uniform(0, 1) > switch_prob:
            return

        profiles = list(Member._STRATEGY_PROFILES.keys())
        logits = np.array([scores.get(name, 0.0) for name in profiles], dtype=float)

        if island_context:
            profile_shares = island_context.get("profile_shares")
            profile_entropy = island_context.get("profile_entropy")
            if isinstance(profile_shares, dict) and profile_shares:
                expected = 1.0 / len(profile_shares)
                diversity_weight = Member._PROFILE_DIVERSITY_WEIGHT
                if profile_entropy is not None:
                    diversity_weight *= float(np.clip(1.0 - profile_entropy, 0.0, 1.0))
                for idx, name in enumerate(profiles):
                    share = float(profile_shares.get(name, 0.0))
                    rarity = (expected - share) / max(expected, 1e-6)
                    logits[idx] += diversity_weight * rarity

        logits = logits - np.max(logits)
        temperature = max(Member._PROFILE_TEMPERATURE, 1e-6)
        weights = np.exp(logits / temperature)
        total = float(np.sum(weights))
        if total <= 0:
            probs = np.ones(len(profiles)) / len(profiles)
        else:
            probs = weights / total

        new_profile = self._rng.choice(profiles, p=probs)
        if new_profile != self.strategy_profile:
            self.strategy_profile = new_profile
            self._apply_profile_scaling()
            state["profile_tenure"] = 0

    def _interaction_alignment(self, other: "Member") -> float:
        benefit = (
            self.interaction_memory["benefit_received"].get(other.id, 0.0)
            + 2.0 * self.interaction_memory["land_received"].get(other.id, 0.0)
        )
        harm = self.interaction_memory["attack_received"].get(other.id, 0.0)
        base_score = np.tanh((benefit - harm) / Member._RELATION_SCORE_SCALE)

        color_score = 0.0
        if np.linalg.norm(np.array(self._current_color) - np.array(other._current_color)) < 30:
            color_score = 0.2

        return base_score + color_score

    def _target_scale(self, decision_name: str, other: Optional["Member"]) -> float:
        if other is None:
            return 1.0
        alignment = self._interaction_alignment(other)

        if decision_name == "attack":
            scale = 1.0 - 0.3 * alignment
        elif decision_name in ("offer", "offer_land", "reproduce", "clear"):
            scale = 1.0 + 0.3 * alignment
        else:
            scale = 1.0

        return float(np.clip(scale, *Member._TARGET_SCALE_RANGE))

    def _effective_parameters(
        self,
        decision_name: str,
        island: "Island.Island",
        other: Optional["Member"] = None,
    ) -> np.ndarray:
        base = self._ensure_parameter_length(decision_name)
        profile_scale_applied = self.strategy_state.get("profile_scale_applied")
        if profile_scale_applied is None:
            profile_scale_applied = True
        decision_scale = 1.0 if profile_scale_applied else (
            self.strategy_state.get("decision_scale", {}).get(decision_name, 1.0)
        )
        scale = (
            decision_scale
            * self.strategy_state["adaptive_scale"].get(decision_name, 1.0)
            * self._target_scale(decision_name, other)
        )
        scale = float(np.clip(scale, *Member._DECISION_SCALE_RANGE))
        return self._clip_parameters(decision_name, base * scale)

    def apply_reward_update(self, decision_name: str, input_vector: List[float], reward: float) -> None:
        params = self._ensure_parameter_length(decision_name)
        baseline = float(self.strategy_state.get("reward_baseline", 0.0))
        advantage = reward - baseline
        update = Member._LEARNING_RATE * advantage * np.array(input_vector)
        self.parameter_dict[decision_name] = self._clip_parameters(decision_name, params + update)
        
# ##############################################################################
# ##################################### 状态 ####################################
    
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
    
    @property
    def overall_productivity(self) -> float:
        """
        基于土地的总产量
        """
        return self.productivity * (self.land_num / Member._STD_LAND)**Member._PRODUCE_ELASTICITY

    def autopsy(self) -> bool:
        """验尸，在Member类中结算死亡，返回是否死亡"""
        if self.vitality <= 0:
            self.vitality = 0
            return True
        return False

    @property
    def is_qualified_to_reproduce(self):
        return (
            self.age >= Member._MIN_REPRODUCE_AGE
            and
            self.land_num >= Member._LAND_HERITAGE + 1
        )

    def center_of_land(self, land: Land.Land) -> np.ndarray:
        """
        估计土地的中心，方法：
        计算自己拥有的每一块土地到自己其他土地的“最短距离”，
        对于每一块土地，计算他离其他所有土地的距离之和，
        距离之和最短的土地，即为大致的中心
        """
        
        dis_mat = np.zeros((self.land_num, self.land_num))
        for idx_1, land_1 in enumerate(self.owned_land):
            for idx_2, land_2 in enumerate(self.owned_land):
                if idx_1 < idx_2:
                    dis_mat[idx_1, idx_2] = land.distance(land_1, land_2)**2
                    dis_mat[idx_2, idx_1] = dis_mat[idx_1, idx_2]
        
        total_dis = np.sum(dis_mat, axis=0)
        approxed_center_id = np.argmin(total_dis)

        return np.array(self.owned_land[approxed_center_id])

    #########################################################################
    ################################## 动作 ##################################
    @property
    def concat_param_dict(self) -> Dict[str, Dict[str, float]]:
        concat_dict = {}
        for key, values in self.parameter_dict.items():
            concat_dict[key] = dict(zip(self._DECISION_INPUT_NAMES, values))
        
        return concat_dict

    def _generate_decision_inputs(
        self, 
        object: Optional["Member"], 
        island: Island.Island,
        # normalize: bool = True
        ) -> Dict:

        input_dict = {}

        input_dict["self_productivity"] = self.overall_productivity / Member._MAX_PRODUCTIVITY
        input_dict["self_vitality"] = self.vitality / Member._MAX_VITALITY
        input_dict["self_cargo"] = np.tanh(self.cargo * Member._CARGO_SCALE)
        input_dict["self_age"] = self.age / Member._MAX_AGE
        input_dict["self_neighbor"] = len(self.current_clear_list) / Member._MAX_NEIGHBOR

        if object is None:
            input_dict["obj_productivity"] = 0.0
            input_dict["obj_vitality"] = 0.0
            input_dict["obj_cargo"] = 0.0
            input_dict["obj_age"] = 0.0
            input_dict["obj_neighbor"] = 0.0

            input_dict["victim_overlap"] = 0.0
            input_dict["benefit_overlap"] = 0.0
            input_dict["benefit_land_overlap"] = 0.0

            input_dict["victim_passive"] = 0.0
            input_dict["victim_active"] = 0.0
            input_dict["benefit_passive"] = 0.0
            input_dict["benefit_active"] = 0.0
            input_dict["benefit_land_passive"] = 0.0
            input_dict["benefit_land_active"] = 0.0

            input_dict["recent_help_received"] = 0.0
            input_dict["recent_harm_received"] = 0.0
            return input_dict

        assert self is not object, "决策函数中主体和对象不能相同"

        input_dict["obj_productivity"] = object.overall_productivity / Member._MAX_PRODUCTIVITY
        input_dict["obj_vitality"] = object.vitality / Member._MAX_VITALITY
        input_dict["obj_cargo"] = np.tanh(object.cargo * Member._CARGO_SCALE)
        input_dict["obj_age"] = object.age / Member._MAX_AGE
        input_dict["obj_neighbor"] = len(object.current_clear_list) / Member._MAX_NEIGHBOR

        (
            input_dict["victim_overlap"], 
            input_dict["benefit_overlap"], 
            input_dict["benefit_land_overlap"]
        ) = island._overlap_of_relations(self, object)
        
        (
            input_dict["victim_passive"], 
            input_dict["victim_active"], 
            input_dict["benefit_passive"], 
            input_dict["benefit_active"],
            input_dict["benefit_land_passive"], 
            input_dict["benefit_land_active"],
        ) = island._relations_w_normalize(self, object)

        recent_help = (
            self.interaction_memory["benefit_received"].get(object.id, 0.0)
            + 2.0 * self.interaction_memory["land_received"].get(object.id, 0.0)
        )
        recent_harm = self.interaction_memory["attack_received"].get(object.id, 0.0)
        input_dict["recent_help_received"] = np.tanh(
            recent_help / Member._RELATION_SCORE_SCALE
        )
        input_dict["recent_harm_received"] = np.tanh(
            recent_harm / Member._RELATION_SCORE_SCALE
        )

        return input_dict

    def decision(
        self, 
        decision_name: str, 
        object: Optional["Member"],
        island: Island.Island,
        threshold: float = 1,
        backend: Optional[str] = None,
        logger = None,
    ) -> bool:
        if object is None and decision_name != "clear":
            raise ValueError("决策需要目标成员，除非是clear动作。")
        input_dict = self._generate_decision_inputs(object, island)
        input_vector = [input_dict[para_name] for para_name in Member._DECISION_INPUT_NAMES]
        effective_params = self._effective_parameters(decision_name, island, object)

        if backend is None:
            backend = self._DECISION_BACKEND

        if backend == "inner product":
            inner = np.sum(effective_params * input_vector)
            decision = inner > threshold
            short_reason = "内积决策"

        elif backend == "gemini":
            decision, short_reason = decision_using_gemini(
                decision_name, 
                input_dict, 
                effective_params
            )
        
        elif backend == "gpt3.5":
            decision, short_reason = decision_using_gpt35(
                decision_name, 
                input_dict, 
                effective_params
            )
        
        else:
            raise ValueError(f"未知的决策后端: {self._DECISION_BACKEND}")
        
        if logger is not None:
            if decision:
                decision_str = ""
            else:
                decision_str = "不"
            logger.info(f"{self}{decision_str}决定对{object}：{decision_name}，原因：{short_reason}")

        return decision

    def parameter_absorb(
        self,
        contributor_list: List[Member] = [],
        weight_list: List[float] = [],
        fluctuation_amplitude = 0,
    ) -> None:
        """产生多个人的决策参数加权平均值"""
        contr_num = len(contributor_list)
        if weight_list == []:
            weight_list = np.ones(contr_num) / contr_num

        # 加权平均
        new_dict = {
            key: contributor_list[0]._ensure_parameter_length(key).copy() * weight_list[0]
            for key in contributor_list[0].parameter_dict.keys()
        }

        for idx in range(1, len(contributor_list)):
            contributor = contributor_list[idx]
            for key in new_dict.keys():
                new_dict[key] += contributor._ensure_parameter_length(key) * weight_list[idx]

        # 浮动
        if fluctuation_amplitude > 0:
            for key in new_dict.keys():
                new_dict[key] += self._rng.uniform(
                    - fluctuation_amplitude, 
                    + fluctuation_amplitude,
                    size = new_dict[key].size
                )
        self.parameter_dict = new_dict

    def acquire_land(self, land_location: Tuple[int, int]):
        self.owned_land.append(land_location)
        self.land_num += 1

    def discard_land(self, land_location: Tuple[int, int]):
        assert land_location in self.owned_land, "只能丢弃拥有的土地"

        self.owned_land.remove(land_location)
        self.land_num -= 1
    
    def produce(self) -> float:
        """生产，将收获装入cargo"""
        self.cargo += self.overall_productivity

        return self.overall_productivity

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
            "land_num": [self.land_num],
            "color_R": [self._color[0]],
            "color_G": [self._color[1]],
            "color_B": [self._color[2]],
            "current_R": [self._current_color[0]],
            "current_G": [self._current_color[1]],
            "current_B": [self._current_color[2]],
        }, index=[self.id])

        land_x = ""
        land_y = ""
        for x, y in self.owned_land:
            land_x = land_x + f"{x:d} "
            land_y = land_y + f"{y:d} "
        info_df["land_x"] = [land_x]
        info_df["land_y"] = [land_y]
        
        # if self.parent_1 is not None:
        #     info_df["parent_1"] = [self.parent_1.id]
        # if self.parent_2 is not None:
        #     info_df["parent_2"] = [self.parent_2.id]
        # children_str = ""
        # for child in self.children:
        #     children_str = children_str + f"{child.id} "
        # info_df["children"] = [children_str]

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

    # def load_from_row(self, row):
