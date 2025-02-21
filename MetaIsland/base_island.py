from __future__ import annotations
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from time import time
import logging
import pickle
import sys
from collections import defaultdict
from termcolor import colored

from utils.save import path_decorator

if TYPE_CHECKING:
    from MetaIsland.base_member import BaseMember
    from MetaIsland.base_land import BaseLand

class BaseIsland:
    _MIN_MAX_INIT_RELATION = {
        "victim": [-500, 1000],                # Negative values initialize to 0
        "benefit": [-500, 1000],        
        "benefit_land": [-30, 30],           
    }

    _NEIGHBOR_SEARCH_RANGE = 1000
    _REPRODUCE_REQUIREMENT = 100                            
    _RECORD_PERIOD = 1

    def __init__(
        self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
    ) -> None:
        # Import here to avoid circular imports
        from MetaIsland.base_member import BaseMember
        from MetaIsland.base_land import BaseLand
        from MetaIsland.settings import name_list

        # Set and record random seed
        self._create_from_file = False
        self._file_name = ""
        self._random_seed = int(random_seed) if random_seed is not None else int(time())
        self._rng = np.random.default_rng(self._random_seed)

        # Save path and logging setup
        self._save_path = path_decorator(save_path)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'{save_path}/log.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Initialize members
        self._NAME_LIST = self._rng.permutation(name_list)
        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num
        
        # Create initial members
        self.init_members = [BaseMember(self._NAME_LIST[i], id=i, surviver_id=i, rng=self._rng) 
                           for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # Initialize land
        assert land_shape[0] * land_shape[1] > init_member_number, "Land area should be greater than initial population"
        self.land = BaseLand(land_shape)
        
        # Assign initial land
        _loc_idx_list = self._rng.choice(
            range(land_shape[0] * land_shape[1]),
            size=self.init_member_num,
            replace=False,
        )
        _loc_list = [(int(loc_idx / land_shape[0]), loc_idx % land_shape[0]) 
                     for loc_idx in _loc_idx_list]
        for idx in range(self.init_member_num):
            self._acquire_land(self.all_members[idx], _loc_list[idx])

        # Initialize relationship matrices
        self.relationship_dict = {
            'victim': None,
            'benefit': None,
            'benefit_land': None
        }

        for key in self.relationship_dict:
            min_val, max_val = BaseIsland._MIN_MAX_INIT_RELATION[key]
            rela = self._rng.uniform(min_val, max_val, size=(self.init_member_num, self.init_member_num))
            np.fill_diagonal(rela, np.nan)
            self.relationship_dict[key] = rela

        # Initialize round counter
        self.current_round = 0

        # Initialize record keeping
        self.record_action_dict = {
            'attack': {},
            'benefit': {},
            'benefit_land': {}
        }
        self.record_total_dict = {
            'attack': [],
            'benefit': [],
            'benefit_land': []
        }
        self.record_total_consumption = []
        self.record_total_production = []
        self.record_born = []
        self.record_death = []

    def _backup_member_list(self, member_list: List['BaseMember']) -> List['BaseMember']:
        """Copy member_list"""
        return [member for member in member_list]

    def _acquire_land(self, member: 'BaseMember', location: Tuple[int, int]) -> None:
        """Give land to a member"""
        assert self.land[location] is None, "Land must be unowned"
        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = member
        member.acquire_land(location)

    def _discard_land(self, member: 'BaseMember', location: Tuple[int, int]) -> None:
        """Remove land from a member"""
        assert location in member.owned_land, "Can only discard owned land"
        assert self.land[location] == member, "Can only discard own land"
        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = None
        member.discard_land(location)

    def get_neighbors(self) -> None:
        """Update neighbor lists for all members"""
        for member in self.current_members:
            self._get_neighbors(member)

    def _get_neighbors(self, member: 'BaseMember') -> None:
        """Get neighbors for a single member"""
        (
            member.current_clear_list,
            member.current_self_blocked_list,
            member.current_neighbor_blocked_list,
            member.current_empty_loc_list,
        ) = self.land.neighbors(
            member,
            self,
            BaseIsland._NEIGHBOR_SEARCH_RANGE
        )

    def save_to_pickle(self, file_name: str) -> None:

        sys.setrecursionlimit(500000)

        file = open(file_name, 'wb') 
        pickle.dump(self, file)
    
    def new_round(self, save_file: bool = True, log_status=False):
        # 输出内容
        if self.current_round % BaseIsland._RECORD_PERIOD == 0:
            # 保存
            if save_file:
                self.save_to_pickle(self._save_path + f"{self.current_round:d}.island")

            # 输出
            if log_status:
                self.log_status()

            # 初始化存储
            self._record_init_per_period()

        # 回合数+1
        self.current_round += 1

        # 每个存活成员增加一岁
        for member in self.current_members:
            member.age += 1
        
    def produce(self) -> None:
        """Have all members produce resources"""
        for member in self.current_members:
            member.produce()

    def consume(self) -> None:
        """Have all members consume resources"""
        for member in self.current_members:
            consumption = member.consume()
            if member.autopsy():
                self.declare_dead(member)
        
        for member in self.current_members:
            member.recover()

    def declare_dead(self, member: 'BaseMember') -> None:
        """Remove a dead member"""
        for loc in member.owned_land.copy():
            self._discard_land(member, loc)
        self.member_list_modify(drop=[member])
        self._logger.info(f"{member.name} died.")

    def member_list_modify(self, append: List['BaseMember'] = [], drop: List['BaseMember'] = []) -> None:
        """Modify the member lists"""
        if drop:
            self._member_list_drop(drop)

    def _member_list_drop(self, drop: List['BaseMember'] = []) -> None:
        """Remove members from lists"""
        drop_id = np.array([member.id for member in drop])
        drop_sur_id = np.array([member.surviver_id for member in drop])

        if (drop_sur_id == None).any():
            raise AttributeError("Dropped members must have surviver_id")

        for member in drop:
            assert member.owned_land == [], "Dropped members should have no land"

        argsort_sur_id = np.argsort(drop_sur_id)[::-1]
        drop_id = drop_id[argsort_sur_id]
        drop_sur_id = drop_sur_id[argsort_sur_id]
        
        for idx in range(len(drop_id)):
            id_to_drop = drop_id[idx]
            sur_id_to_drop = drop_sur_id[idx]
            assert self.current_members[sur_id_to_drop].id == id_to_drop

            self.current_members[sur_id_to_drop].surviver_id = None
            del self.current_members[sur_id_to_drop]
            self.current_member_num -= 1

        for key in self.relationship_dict:
            tmp = np.delete(self.relationship_dict[key], drop_sur_id, axis=0)
            tmp = np.delete(tmp, drop_sur_id, axis=1)
            self.relationship_dict[key] = tmp

        for sur_id in range(self.current_member_num):
            self.current_members[sur_id].surviver_id = sur_id

    def _offer(self, member_1: 'BaseMember', member_2: 'BaseMember', parameter_influence: float) -> None:
        """Member 1 offers resources to Member 2"""
        if parameter_influence <= 0:
            return
            
        # Transfer resources
        member_1.cargo -= parameter_influence
        member_2.cargo += parameter_influence
        
        # Update relationship matrix
        m1_idx = member_1.surviver_id
        m2_idx = member_2.surviver_id
        self.relationship_dict['benefit'][m1_idx][m2_idx] += parameter_influence

    def _offer_land(self, member_1: 'BaseMember', member_2: 'BaseMember', parameter_influence: float) -> None:
        """Member 1 offers land to Member 2"""
        if parameter_influence <= 0 or not member_1.owned_land:
            return
            
        # Select random land to transfer
        land_loc = self._rng.choice(list(member_1.owned_land))
        self._discard_land(member_1, land_loc)
        self._acquire_land(member_2, land_loc)
        
        # Update relationship matrix
        m1_idx = member_1.surviver_id
        m2_idx = member_2.surviver_id
        self.relationship_dict['benefit_land'][m1_idx][m2_idx] += 1

    def _attack(self, member_1: 'BaseMember', member_2: 'BaseMember') -> None:
        """Member 1 attacks Member 2"""
        damage = member_1.strength
        member_2.vitality -= damage
        
        # Update relationship matrix
        m1_idx = member_1.surviver_id
        m2_idx = member_2.surviver_id
        self.relationship_dict['victim'][m2_idx][m1_idx] += damage

    def _bear(self, member_1: 'BaseMember', member_2: 'BaseMember') -> None:
        """Members reproduce to create a new member"""
        if not member_1.is_qualified_to_reproduce or not member_2.is_qualified_to_reproduce:
            return
            
        # Create new member
        new_id = len(self.all_members)
        new_member = BaseMember.born(member_1, member_2, self._NAME_LIST[new_id], new_id, self.current_member_num, self._rng)
        
        # Add to member lists
        self.all_members.append(new_member)
        self.current_members.append(new_member)
        self.current_member_num += 1
        
        # Expand relationship matrices
        for key in self.relationship_dict:
            old_matrix = self.relationship_dict[key]
            new_size = len(self.current_members)
            new_matrix = np.zeros((new_size, new_size))
            new_matrix[:-1, :-1] = old_matrix
            np.fill_diagonal(new_matrix, np.nan)
            self.relationship_dict[key] = new_matrix

        # Transfer land from parents
        for parent in [member_1, member_2]:
            for _ in range(BaseMember._LAND_HERITAGE):
                if parent.land_num <= BaseMember._LAND_HERITAGE:
                    break
                land_loc = self._rng.choice(list(parent.owned_land))
                self._discard_land(parent, land_loc)
                self._acquire_land(new_member, land_loc)

    def _expand(self, member_1: 'BaseMember', member_2: 'BaseMember') -> None:
        """Members cooperate to expand into new territory"""
        # Find available adjacent land
        empty_locations = []
        for loc in member_1.current_empty_loc_list:
            if loc in member_2.current_empty_loc_list:
                empty_locations.append(loc)
                
        if not empty_locations:
            return
            
        # Choose random empty location and give to member_1
        new_loc = self._rng.choice(empty_locations)
        self._acquire_land(member_1, new_loc)

    def log_status(
        self,
        action = False,
        summary = True,
        members = True,
        log_instead_of_print = True,
    ):
        log_str = ""

        log_str += "#" * 21 + f" {self.current_round:d} " + "#" * 21 + "\n"

        if action:
            log_str += "=" * 21 + " 攻击 " + "=" * 21 + "\n"
            if self.record_action_dict["attack"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["attack"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

            log_str += "=" * 21 + " 给予 " + "=" * 21 + "\n"
            if self.record_action_dict["benefit"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n" 

            log_str += "=" * 20 + " 给予土地 " + "=" * 20 + "\n"
            if self.record_action_dict["benefit_land"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit_land"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

        if summary:
            log_str += "=" * 50 + "\n"
            log_str += f"本轮出生：{self.record_born} \n"
            log_str += f"本轮死亡：{self.record_death} \n"
            log_str += f"本轮总给予：{self.record_total_dict['benefit'][-1]:.1f} \n"
            log_str += f"本轮总攻击：{self.record_total_dict['attack'][-1]:.1f} \n"
            log_str += f"本轮总产量：{self.record_total_production[-1]:.1f} \n"
            log_str += f"本轮总消耗：{self.record_total_consumption[-1]:.1f} \n"
            
        if members:
            log_str += "=" * 50 + "\n"
            log_str += "\t ID Sur_ID  姓名          年龄   血量    仓库    土地数 \n"
            for member in self.current_members:
                space_after_name = " " * (10 - len(member.name))

                mem_str = (
                    f"\t[{member.id}, {member.surviver_id}] "
                    f"{member.name}:{space_after_name}"
                    f"   {member.age}," 
                    f"   {member.vitality:.1f},"
                    f"   {member.cargo:.1f}"
                    f"   {member.land_num:d}({100*member.land_num/np.prod(self.land.shape):.1f}%)"
                    "\n"
                )
            
                log_str += mem_str

        if log_instead_of_print:
            self._logger.info(log_str)
        else:
            print(log_str) 

    def _record_init_per_period(self) -> None:
        """Initialize/reset record keeping for each period"""
        # Reset action records
        for key in self.record_action_dict.keys():
            self.record_action_dict[key] = {}
            
        # Append new entries for total records
        for key in self.record_total_dict.keys():
            self.record_total_dict[key].append(0)

        self.record_total_consumption.append(0)
        self.record_total_production.append(0)

        self.record_born = []
        self.record_death = [] 