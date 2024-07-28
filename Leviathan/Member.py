from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from Leviathan.prompt import decision_using_gemini, decision_using_gpt
import Leviathan.Island as Island
import Leviathan.Land as Land

def colored(rgb, text):
    r, g, b = rgb
    # return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
    return text

class Member():
    # Productivity
    # prod * (land / standard)**0.5
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 15, 20       # Productivity attribute
    _PRODUCE_ELASTICITY = 0.5                           # Productivity elasticity, the power by which productivity increases with land growth
    _STD_LAND = 4                                       # Standard land, at this land, the yield per round equals productivity
    _MAX_VITALITY = 100

    # Decision-making
    _CARGO_SCALE = 0.02                                 # The scaling amount of cargo when calculating the decision function
    _RELATION_SCALES = [0.01, 0.01, 0.25]                  # The scaling amount in the decision function when calculating relationships
    _MAX_NEIGHBOR = 4                                   # Maximum number of neighbors
    _DECISION_BACKEND = "gpt"                        # The backend of the decision function (inner product, gemini, gpt3.5)

    # Initial values
    _INIT_MIN_VIT, _INIT_MAX_VIT = 10, 90             # Initial vitality
    _INIT_MIN_CARGO, _INIT_MAX_CARGO = 0, 100         # Initial food storage
    _INIT_MIN_AGE, _INIT_MAX_AGE = 10, 99           # Initial age
    _CHILD_VITALITY = 50                                # Vitality at birth

    # Consumption-related calculation parameters
    _CONSUMPTION_BASE = 15                              # Base consumption amount
    _MAX_AGE = _INIT_MAX_AGE                            # Theoretical maximum age (the age at which consumption equals maximum vitality)
    _COMSUMPTION_CLIMBING_AGE = int(0.5 * _MAX_AGE)     # The age at which consumption starts to increase significantly
    __AGING_EXPOENT = np.log(_MAX_VITALITY - _CONSUMPTION_BASE) / (_MAX_AGE - _COMSUMPTION_CLIMBING_AGE)

    # Action scale
    _MIN_STRENGTH, _MAX_STRENGTH = 0.1, 0.3             # Attack power as a ratio of current vitality
    _MIN_STEAL, _MAX_STEAL = 0.1, 0.3                   # Stealing value as a ratio of current vitality
    _MIN_OFFER_PERCENTAGE, _MAX_OFFER_PERCENTAGE = 0.1, 0.3                   # Offering value as a ratio of current cargo
    _MIN_OFFER = 5                                      # Minimum offering value

    # Reproduction
    _MIN_REPRODUCE_AGE = int(0.18 * _MAX_AGE)           # Minimum age
    _PARAMETER_FLUCTUATION = 0.4                       # Parameter inheritance fluctuation
    _LAND_HERITAGE = np.ceil(_STD_LAND / 2).astype(int) # The amount of land given at reproduction

    # Trade
    _PARAMETER_INFLUENCE = 0.01                       # Parameter influence after trade

    # Names of decision parameters
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

        "victim_overlap",                   # Overlapping part of the relationship network
        "benefit_overlap",
        "benefit_land_overlap",

        # Refined relationship network inner product
        # "victim_passive_passive",           # victim_passive_passive represents the rows owned by self in the victim's memory, multiplied by the rows owned by obj
        # "victim_passive_active",            # The first p/a represents self's memory, the second p/a represents obj's memory
        # "victim_active_passive",            # passive - rows, active - columns
        # "victim_active_active",
        # "benefit_passive_passive",
        # "benefit_passive_active",
        # "benefit_active_passive",
        # "benefit_active_active",

        "victim_passive",                   # self's memory of obj
        "victim_active",                    # obj's memory of self
        "benefit_passive",
        "benefit_active",
        "benefit_land_passive",
        "benefit_land_active",
    ]
    _DECISION_NAMES = [
        "attack",
        "offer",
        "reproduce",
        "clear",
        "offer_land",
    ]
    _parameter_name_dict = {}               # Names of parameters
    for key in _DECISION_NAMES:
        _parameter_name_dict[key] = []
        for name in _DECISION_INPUT_NAMES:
            _parameter_name_dict[key].append(key + "_" + name)

    # Initial decision parameters, each row of the list represents various parameters, columns represent minimum and maximum values
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

        # Random number generator
        self._rng = rng

        # Name
        self.name = name
        self.id = id
        self.surviver_id = surviver_id      # ID in Island.current_members

        # Family list
        # self.parent_1 = None
        # self.parent_2 = None
        # self.children = []

        # Character color
        self._color = [0, 0, 0]
        while np.std(self._color) < 100:
            self._color = np.round(self._rng.uniform(0, 255, size=3)).astype(int)
        self._current_color = self._color.copy()

        # Territory
        self.owned_land: List[Tuple[int, int]] = []
        self.land_num = 0

        self.current_clear_list = []
        self.current_self_blocked_list = []
        self.current_neighbor_blocked_list = []
        self.current_empty_loc_list = []

        # Production-related attributes and states
        self.productivity = self._rng.uniform(Member._MIN_PRODUCTIVITY, Member._MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(Member._INIT_MIN_VIT, Member._INIT_MAX_VIT)
        self.cargo = self._rng.uniform(Member._INIT_MIN_CARGO, Member._INIT_MAX_CARGO)
        self.age = int(self._rng.uniform(Member._INIT_MIN_AGE, Member._INIT_MAX_AGE))

        # Randomly initialize decision parameters
        self.parameter_dict = {}
        for des_name in Member._DECISION_NAMES:
            para_range = Member._INITIAL_PRAMETER[des_name]
            self.parameter_dict[des_name] = (
                self._rng.uniform(0, 1, size=len(Member._DECISION_INPUT_NAMES))
                * (para_range[:, 1] - para_range[:, 0])
                + para_range[:, 0]
            )

    def __str__(self):
        """Override print function representation"""
        return colored(self._current_color, f"{self.name}({self.id})")

    def __repr__(self):
        """Override other print form representation"""
        return self.__str__()

    # def __getstate__(self):
    #     # print(f"Pickling {self.__class__.__name__} object with ID: {self.__dict__}")
    #     state = self.__dict__.copy()
    #     # Add any custom pickling logic here
    #     return state
        
# ##############################################################################
# ##################################### State ####################################
    
    @property
    def strength(self) -> float:
        """Combat power: damage caused by each attack"""
        return (
            self._rng.uniform(Member._MIN_STRENGTH, Member._MAX_STRENGTH)
            * self.vitality
        )
    
    @property
    def steal(self) -> float:
        """Harvest from each theft"""
        return (
            self._rng.uniform(Member._MIN_STEAL, Member._MAX_STEAL) 
            * self.vitality
        )

    @property
    def offer(self) -> float:
        """Amount given each time"""
        return (
            self._rng.uniform(Member._MIN_OFFER_PERCENTAGE, Member._MAX_OFFER_PERCENTAGE) 
            * self.cargo
        )

    @property
    def consumption(self) -> float:
        """Consumption amount per round"""
        amount = (Member._CONSUMPTION_BASE 
            + np.exp(Member.__AGING_EXPOENT * (self.age - Member._COMSUMPTION_CLIMBING_AGE))
        )
        return amount 
    
    @property
    def overall_productivity(self) -> float:
        """
        Total productivity based on land
        """
        return self.productivity * (self.land_num / Member._STD_LAND)**Member._PRODUCE_ELASTICITY

    def autopsy(self) -> bool:
        """Autopsy, settle death in the Member class, return whether dead"""
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
        Estimate the center of the land, method:
        Calculate the "shortest distance" from each piece of land owned to other lands,
        For each piece of land, calculate the sum of distances to all other lands,
        The land with the shortest distance sum is the approximate center
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
    ################################## Actions ##################################
    @property
    def concat_param_dict(self) -> Dict[str, Dict[str, float]]:
        concat_dict = {}
        for key, values in self.parameter_dict.items():
            concat_dict[key] = dict(zip(self._DECISION_INPUT_NAMES, values))
        
        return concat_dict

    def _generate_decision_inputs(
        self, 
        object: Member, 
        island: Island.Island,
        # normalize: bool = True
        ) -> Dict:

        assert self is not object, "The subject and object in the decision function cannot be the same"

        input_dict = {}

        input_dict["self_productivity"] = self.overall_productivity / Member._MAX_PRODUCTIVITY
        input_dict["self_vitality"] = self.vitality / Member._MAX_VITALITY
        input_dict["self_cargo"] = np.tanh(self.cargo * Member._CARGO_SCALE)
        input_dict["self_age"] = self.age / Member._MAX_AGE
        input_dict["self_neighbor"] = len(self.current_clear_list) / Member._MAX_NEIGHBOR

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

        return input_dict

    def decision(
        self, 
        decision_name: str, 
        object: Member,
        island: Island.Island,
        threshold: float = 1,
        backend: Optional[str] = None,
        logger = None,
    ) -> bool:
        input_dict = self._generate_decision_inputs(object, island)

        lang_map = {
            "en": "English",
            "cn": "中文",
            "jp": "日本語",
            "sp": "Español"
        }
        log_lang = lang_map.get(island.log_lang, "en")  # Default to English if not found
        
        if backend is None:
            backend = self._DECISION_BACKEND

        if backend == "inner product":
            input = [input_dict[para_name] for para_name in Member._DECISION_INPUT_NAMES]
            inner = np.sum(self.parameter_dict[decision_name] * input)
            decision = inner > threshold
            short_reason = "Inner product decision"

        elif backend == "gemini":
            decision, short_reason = decision_using_gemini(
                decision_name, 
                input_dict, 
                self.parameter_dict[decision_name]
            )
        
        elif backend == "gpt":
            decision, short_reason = decision_using_gpt(
                decision_name, 
                input_dict, 
                self.parameter_dict[decision_name],
                log_lang
            )
        
        else:
            raise ValueError(f"Unknown decision backend: {self._DECISION_BACKEND}")
        
        if logger is not None:
            if decision:
                decision_str = ""
            else:
                decision_str = "not "
            logger.info(f"{self}{decision_str}decided to {object}: {decision_name}, reason: {short_reason}")
            island.decision_record.append(f"{self}{decision_str}decided to {object}: {decision_name}, reason: {short_reason}")

        return decision

    def parameter_absorb(
        self,
        contributor_list: List[Member] = [],
        weight_list: List[float] = [],
        fluctuation_amplitude = 0,
    ) -> None:
        """Generate a weighted average of decision parameters from multiple contributors"""
        contr_num = len(contributor_list)
        if weight_list == []:
            weight_list = np.ones(contr_num) / contr_num

        # Weighted average
        new_dict = {
            key: val.copy() * weight_list[0] 
            for key, val in contributor_list[0].parameter_dict.items()
        }

        for idx in range(1, len(contributor_list)):
            contributor = contributor_list[idx]
            for key in new_dict.keys():
                new_dict[key] += contributor.parameter_dict[key] * weight_list[idx]

        # Fluctuation
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
        assert land_location in self.owned_land, "Can only discard owned land"

        self.owned_land.remove(land_location)
        self.land_num -= 1
    
    def produce(self) -> float:
        """Produce and store the harvest in cargo"""
        self.cargo += self.overall_productivity

        return self.overall_productivity

    def consume(self) -> float:
        """Consume vitality"""
        consumption = self.consumption
        if self.vitality > consumption:
            self.vitality -= self.consumption
        else:
            consumption = self.vitality
            self.vitality = 0
        return consumption

    def recover(self) -> None:
        """Use cargo to recover vitality"""
        amount = np.min([self.cargo, Member._MAX_VITALITY - self.vitality])
        self.vitality += amount
        self.cargo -= amount

# ================================== Save =======================================
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

        # Store decision parameters
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


