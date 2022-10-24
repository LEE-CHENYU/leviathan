from __future__ import annotations
import numpy as np

from Member import Member
import Island

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


class Land():
    def __init__(
        self,
        shape: Tuple[int, int],

    ):
        self.shape = shape
        self.owner: List[List[Member]] = []
        for _ in range(shape[0]):
            self.owner.append([None] * self.shape[1])

    def __getitem__(self, key) -> Member:
        if isinstance(key, tuple):
            return self.owner[key[0]][key[1]]
        else:
            raise ValueError("土地目前只接受元组查询")

    def __str__(self):
        """重载print函数表示"""
        return self.owner.__str__()

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.owner.__repr__()

    def _find_neighbors(
        self, 
        clear_list: List[Member],
        self_blocked_list: List[Member],
        neighbor_blocked_list: List[Tuple(Member, Member)],
        empty_loc_list: List[Tuple[int, int]],
        location: Tuple[int, int], 
        member: Member, 
        is_passed: np.ndarray,
        iteration_cnt: int,
        island: Island.Island,
    ) -> None:

        is_passed[location] = 1

        loc_0, loc_1 = location
        land_owner = self[location]
        in_owned_land = (member == land_owner)

        loc_to_pass = [
            [loc_0, (loc_1 + 1) % self.shape[1]],
            [(loc_0 + 1) % self.shape[0], loc_1],
            [loc_0, (loc_1 - 1) % self.shape[1]],
            [(loc_0 - 1) % self.shape[0], loc_1],
        ]
        island._rng.shuffle(loc_to_pass, axis=0)

        # 遍历四个方向
        for direction in loc_to_pass:
            direction = tuple(direction)
            if not is_passed[direction[0], direction[1] % self.shape[1]]:
                member_to_pass = self[direction]

                # 如果是边界，记录，并继续遍历四个方向
                if member_to_pass is None:
                    empty_loc_list.append(direction)
                    continue

                # 如果成员曾经阻拦
                if member_to_pass in self_blocked_list:
                    continue
                if (land_owner, member_to_pass) in neighbor_blocked_list:
                    if in_owned_land:
                        neighbor_blocked_list.remove((land_owner, member_to_pass))
                        self_blocked_list.append(member_to_pass)
                    continue

                # 如果成员曾经放行，或遇到自己领地
                if member_to_pass in clear_list or member_to_pass == member:
                    self._find_neighbors(
                        clear_list,
                        self_blocked_list,
                        neighbor_blocked_list,
                        empty_loc_list,
                        direction, 
                        member, 
                        is_passed,
                        iteration_cnt + 1,
                        island
                    )
                    continue

                if member_to_pass.decision(
                    "clear", 
                    member,
                    island
                ):
                    clear_list.append(member_to_pass)
                    self._find_neighbors(
                        clear_list,
                        self_blocked_list,
                        neighbor_blocked_list,
                        empty_loc_list,
                        direction, 
                        member, 
                        is_passed,
                        iteration_cnt + 1,
                        island
                    )
                    continue
                else:
                    if in_owned_land:
                        self_blocked_list.append(member_to_pass)
                    else:
                        neighbor_blocked_list.append((land_owner, member_to_pass))
                    continue

    def neighbors(
        self, 
        member: Member, 
        island: Island
    ):
        """
        返回四个列表：
        - clear_list: 允许通行
        - self_blocked_list: 与member直接接壤
        - neighbor_blocked_list: 与member间接接壤的成员以及作为桥梁的地主，存储格式为（地主，间接接壤成员）
        - empty_loc_list: 闲置土地
        """
        clear_list = []
        self_blocked_list = []
        neighbor_blocked_list = []
        empty_loc_list = []

        is_passed = np.zeros(self.shape)

        for land in member.owned_land:
            if is_passed[land]:
                continue
            self._find_neighbors(
                clear_list,
                self_blocked_list,
                neighbor_blocked_list,
                empty_loc_list,
                location = land,
                member = member,
                is_passed = is_passed,
                iteration_cnt = 0,
                island = island
            )

        return (
            clear_list, 
            self_blocked_list,
            neighbor_blocked_list,
            empty_loc_list
        )

    def owner_id(
        self,
    ) -> np.ndarray:
        data = np.zeros(self.shape, dtype=int)
        for idx, _ in np.ndenumerate(data):
            if self[idx] is not None:
                data[idx] = self[idx].id
            else:
                data[idx] = -1

        return data