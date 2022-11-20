from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from Member import Member
import Island

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


class Land():
    def __init__(
        self,
        shape: Tuple[int, int],

    ):
        self.shape = shape
        self.owner: np.ndarray[Member] = np.ndarray(shape, dtype=Member)

    def __getitem__(self, key) -> Member:
        if isinstance(key, tuple):
            return self.owner[key]
        else:
            raise ValueError("土地目前只接受元组查询")

    def __str__(self):
        """重载print函数表示"""
        return self.owner.__str__()

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.owner.__repr__()

    def distance(self, loc_1: Tuple[int, int], loc_2: Tuple[int, int]) -> float:
        dis_in_period = lambda x1, x2, period: np.min([
            (x1 - x2) % period,
            (x2 - x1) % period,
        ])
        direction = [
            dis_in_period(loc_1[0], loc_2[0], self.shape[0]),
            dis_in_period(loc_1[1], loc_2[1], self.shape[1]),
        ]
        distance = np.sqrt(np.linalg.norm(direction))
        return distance

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

    def _color_map(
        self, 
        original_color = True,
        ):
        map = np.zeros(self.shape + (3,), dtype=int)
        for idx in np.ndindex(self.shape):
            mem: Member = self.owner[idx]
            if mem is not None:
                if original_color:
                    map[idx] = mem._color
                else:
                    map[idx] = mem._current_color
            else:
                map[idx] = [255, 255, 255]

        return map

    def _plot_map(
        self,
        ax,
        map
    ):
        ax.imshow(map)

        for (i, j), owner in np.ndenumerate(self.owner):
            if owner is not None:
                ax.text(j, i, f"{owner.surviver_id:d}", ha='center', va='center', c="white", fontsize=5)

    def plot(
        self
    ):
        fig, axs = plt.subplots(1, 2, figsize=np.array(self.shape)[::-1]*0.3, dpi=150)

        ax = axs[0]
        original_map = self._color_map(original_color=True)
        self._plot_map(ax, original_map)

        ax = axs[1]
        current_map = self._color_map(original_color=False)
        self._plot_map(ax, current_map)

        plt.show()