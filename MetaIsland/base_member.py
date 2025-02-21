from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple
from collections import defaultdict

if TYPE_CHECKING:
    from MetaIsland.base_land import BaseLand
    from MetaIsland.base_island import BaseIsland

class BaseMember:
    # Class constants
    _MIN_PRODUCTIVITY, _MAX_PRODUCTIVITY = 15, 20
    _PRODUCE_ELASTICITY = 0.5
    _STD_LAND = 4
    _MAX_VITALITY = 100
    _CONSUMPTION_BASE = 15
    _MAX_AGE = 99
    _LAND_HERITAGE = 2

    def __init__(
        self, 
        name: str, 
        id: int, 
        surviver_id: int,
        rng: np.random.Generator
    ):
        self.name = name
        self.id = id
        self.surviver_id = surviver_id
        self._rng = rng
        
        # Core attributes
        self.productivity = self._rng.uniform(self._MIN_PRODUCTIVITY, self._MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(10, 90)
        self.cargo = self._rng.uniform(0, 100)
        self.age = int(self._rng.uniform(10, self._MAX_AGE))
        
        # Land management
        self.owned_land: List[Tuple[int, int]] = []
        self.land_num = 0
        
        # Neighbor tracking
        self.current_clear_list: List[int] = []
        self.current_self_blocked_list: List[int] = []
        self.current_neighbor_blocked_list: List[Tuple[int, int]] = []
        self.current_empty_loc_list: List[Tuple[int, int]] = []

    @property
    def overall_productivity(self) -> float:
        return self.productivity * (self.land_num / self._STD_LAND)**self._PRODUCE_ELASTICITY

    @property
    def consumption(self) -> float:
        return self._CONSUMPTION_BASE + np.exp(
            np.log(self._MAX_VITALITY - self._CONSUMPTION_BASE) / 
            (self._MAX_AGE - 50) * (self.age - 50)
        )

    @property
    def is_qualified_to_reproduce(self) -> bool:
        return (self.age >= 18 and 
                self.land_num >= self._LAND_HERITAGE + 1 and
                self.vitality >= 50)

    def acquire_land(self, land_location: Tuple[int, int]):
        self.owned_land.append(land_location)
        self.land_num += 1

    def discard_land(self, land_location: Tuple[int, int]):
        self.owned_land.remove(land_location)
        self.land_num -= 1

    def produce(self) -> float:
        production = self.overall_productivity
        self.cargo += production
        return production

    def consume(self) -> float:
        consumed = min(self.vitality, self.consumption)
        self.vitality -= consumed
        return consumed

    def recover(self):
        recovery = min(self.cargo, self._MAX_VITALITY - self.vitality)
        self.vitality += recovery
        self.cargo -= recovery

    def autopsy(self) -> bool:
        return self.vitality <= 0

    @classmethod
    def born(
        cls, 
        parent_1: BaseMember, 
        parent_2: BaseMember,
        name: str,
        id: int,
        surviver_id: int,
        rng: np.random.Generator
    ) -> BaseMember:
        child = cls(name, id, surviver_id, rng)
        child.productivity = (parent_1.productivity + parent_2.productivity) / 2
        child.vitality = 50
        return child
