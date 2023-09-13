from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import Leviathan.Member_mdp as Member_mdp
import Leviathan.Island_mdp as Island_mdp

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


class Land():
    def __init__(
        self,
        shape: Tuple[int, int],

    ):
        self.shape = shape
        self.owner: np.ndarray[Member_mdp.Member] = np.ndarray(shape, dtype=Member_mdp.Member)

    def __getitem__(self, key) -> Member_mdp.Member:
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
        distance = np.linalg.norm(direction)
        return distance

    def _find_neighbors(
        self, 
        clear_list: List[Member_mdp.Member],
        self_blocked_list: List[Member_mdp.Member],
        neighbor_blocked_list: List[Tuple(Member_mdp.Member, Member_mdp.Member)],
        empty_loc_list: List[Tuple[int, int]],
        location: Tuple[int, int], 
        member: Member_mdp.Member, 
        is_passed: np.ndarray,
        iteration_cnt: int,
        max_iter: int,
        island: Island_mdp.Island,
        decision_threshold: int = 1,
    ) -> None:

        if iteration_cnt >= max_iter:
            return

        # 走到这个格子时候剩余的步数
        is_passed[location] = max_iter - iteration_cnt

        # print(f"In inter = {iteration_cnt}, ispassed: ")
        # print("\t", is_passed)

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

            # print(f"\t Now in pos: {direction}, {is_passed[direction[0], direction[1]]}, {max_iter - iteration_cnt - 1}")

            # 如果之前曾经走到过下一个地点，并且当时的剩余步数比现在多，就不用继续探索了
            # if is_passed[direction[0], direction[1]] > (max_iter - iteration_cnt - 1): 

            if is_passed[direction[0], direction[1]] > 0: 
                continue

            member_to_pass = self[direction]

            # 如果是边界，记录，并继续遍历
            if member_to_pass is None:
                if direction not in empty_loc_list:
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
                    max_iter,
                    island
                )
                continue

            if member_to_pass.decision(
                "clear", 
                member,
                island,
                threshold=decision_threshold,
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
                    max_iter,
                    island
                )
                continue
            else:
                if in_owned_land:
                    self_blocked_list.append(member_to_pass)
                else:
                    neighbor_blocked_list.append((land_owner, member_to_pass))
                continue

        return 

    def neighbors(
        self, 
        member: Member_mdp.Member, 
        island: Island_mdp,
        max_iter: int = np.inf,
        decision_threshold: int = 1,
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
            # if is_passed[land]:
            #     continue
            self._find_neighbors(
                clear_list,
                self_blocked_list,
                neighbor_blocked_list,
                empty_loc_list,
                location = land,
                member = member,
                is_passed = is_passed,
                iteration_cnt = 0,
                max_iter = max_iter,
                island = island,
                decision_threshold = decision_threshold,
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
            mem: Member_mdp.Member = self.owner[idx]
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
        map,
        show_id = True,
    ):
        ax.imshow(map)

        if not show_id:
            return

        for (i, j), owner in np.ndenumerate(self.owner):
            if owner is not None:
                ax.text(j, i, f"{owner.surviver_id:d}", ha='center', va='center', c="white", fontsize=5)

    def plot(
        self,
        axs = None,
        show_id = True,
    ):
        if axs is None: 
            fig, axs = plt.subplots(1, 2, figsize=np.array(self.shape)[::-1]*0.3, dpi=150)

        ax = axs[0]
        original_map = self._color_map(original_color=True)
        self._plot_map(ax, original_map, show_id)

        ax = axs[1]
        current_map = self._color_map(original_color=False)
        self._plot_map(ax, current_map, show_id)

