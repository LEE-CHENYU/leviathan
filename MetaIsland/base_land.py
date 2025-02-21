from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from MetaIsland.base_member import BaseMember
    from MetaIsland.base_island import BaseIsland

class BaseLand:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        self.owner: np.ndarray = np.full(shape, None, dtype=object)

    def __getitem__(self, key: Tuple[int, int]) -> Optional[BaseMember]:
        return self.owner[key]

    def distance(self, loc_1: Tuple[int, int], loc_2: Tuple[int, int]) -> float:
        return np.linalg.norm([
            min(abs(loc_1[0]-loc_2[0]), self.shape[0]-abs(loc_1[0]-loc_2[0])),
            min(abs(loc_1[1]-loc_2[1]), self.shape[1]-abs(loc_1[1]-loc_2[1]))
        ])

    def neighbors(
        self, 
        member: BaseMember, 
        island: BaseIsland, 
        max_iter: int = 1000
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        # Simplified neighbor finding logic
        clear_list = []
        self_blocked = []
        neighbor_blocked = []
        empty_locs = []

        for loc in member.owned_land:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    x = (loc[0] + dx) % self.shape[0]
                    y = (loc[1] + dy) % self.shape[1]
                    neighbor = self.owner[x, y]
                    
                    if neighbor is None:
                        empty_locs.append((x, y))
                    elif neighbor.id not in clear_list:
                        clear_list.append(neighbor.id)

        return clear_list, self_blocked, neighbor_blocked, empty_locs

    def owner_id_map(self) -> np.ndarray:
        return np.vectorize(lambda x: x.id if x else -1)(self.owner)
