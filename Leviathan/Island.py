import numpy as np
import pandas as pd
from Leviathan.Member import Member, colored
from Leviathan.Land import Land
# from Leviathan.settings import name_list

from Leviathan.save import path_decorator

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import time
import logging
import pickle 
import dill
import sys
import itertools

from collections import defaultdict

def _requirement_for_reproduction(
    member_1: Member, 
    member_2: Member
) -> bool:
    result = (
        (
            member_1.vitality + member_1.cargo
            + member_2.vitality + member_2.cargo
        ) >= Island._REPRODUCE_REQUIREMENT 
        and 
        member_1.is_qualified_to_reproduce
        and 
        member_2.is_qualified_to_reproduce
    )

    # Display reasons for inability to reproduce (for debugging)
    # if not result:
    #     if (
    #         member_1.vitality + member_1.cargo
    #         + member_2.vitality + member_2.cargo
    #     ) < Island._REPRODUCE_REQUIREMENT :
    #         reason = "Not enough vitality and cargo"

    #     elif not member_1.is_qualified_to_reproduce:
    #         reason = f"{member_1} not qualified to reproduce"

    #     elif not member_2.is_qualified_to_reproduce:
    #         reason = f"{member_2} not qualified to reproduce"

    #     print(f"{member_1} and {member_2} cannot reproduce because: {reason}")

    return result

def _requirement_for_offer(
    member_1: Member, 
    member_2: Member
) -> bool:
    return member_1.cargo * Member._MIN_OFFER_PERCENTAGE >= Member._MIN_OFFER

def _requirement_for_offer_land(
    member_1: Member, 
    member_2: Member
) -> bool:
    return member_1.land_num > 1

class Island():
    _MIN_MAX_INIT_RELATION = {
        "victim": [-50, 100],                # If a negative value is randomly generated, this memory is initialized to 0
        "benefit": [-50, 100],        
        "benefit_land": [-3, 3],           
    }

    _NEIGHBOR_SEARCH_RANGE = 1000
    _REPRODUCE_REQUIREMENT = 100                            # Reproduction condition: the sum of the parents' vitality and cargo must be greater than this value
    assert _REPRODUCE_REQUIREMENT > Member._CHILD_VITALITY

    # Record/output period
    _RECORD_PERIOD = 1

    def __init__(
        self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        log_lang: str = "en",
    ) -> None:

        # Set and record random seed
        self._create_from_file = False
        self._file_name = ""
        if random_seed is not None:
            self._random_seed = int(random_seed)
        else:
            self._random_seed = int(time())
        self._rng = np.random.default_rng(self._random_seed)
        self.log_lang = {
            "English": "en",
            "中文": "cn",
            "日本語": "jp",
            "Español": "sp"
        }[log_lang]
        
        print(self.log_lang)

        # Save path
        # self._save_path = path_decorator(save_path)
        self._save_path = save_path
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'{save_path}/log.txt')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Initial number of members, current number of members
        name_list = [
            "Alice", "Bob", "Cathy", "David", "Eve", "Frank", "Grace", "Henry", 
            "Ivy", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Peter", 
            "Queen", "Rose", "Sam", "Tina", "Ulysses", "Violet", "Will", "Xena", 
            "Yuri", "Zack"
        ]
        self._NAME_LIST = self._rng.permutation(name_list)

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # Initial member list, all member list, current member list
        self.init_members = [Member(self._NAME_LIST[i], id=i, surviver_id=i, rng=self._rng) for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # Map
        assert land_shape[0] * land_shape[1] > init_member_number, "Land area should be greater than initial population"
        self.land = Land(land_shape)
        # Allocate land to initial population
        _loc_idx_list = self._rng.choice(
            a = range(land_shape[0] * land_shape[1]),
            size = self.init_member_num,
            replace = False,
        )
        _loc_list = [(int(loc_idx / land_shape[0]), loc_idx % land_shape[0]) for loc_idx in _loc_idx_list]
        for idx in range(self.init_member_num):
            self._acquire_land(self.all_members[idx], _loc_list[idx])

        # Initial member relationships
        # Relationship matrix M, the j-th row (M[j, :]) represents the passive memory of the j-th subject (injured/given...)
        # To modify (add or reduce) member relationships, you need to modify: self.relationship_dict, Member.DECISION_INPUT_NAMES, Member._generate_decision_inputs()
        self.relationship_dict = {}
        for key, (min, max) in Island._MIN_MAX_INIT_RELATION.items():
            rela = self._rng.uniform(
                min, 
                max, 
                size=(self.init_member_num, self.init_member_num)
            )
            rela[rela < 0] = 0  # If a negative value is randomly generated, this memory is set to 0
            np.fill_diagonal(rela, np.nan)

            self.relationship_dict[key] = rela

        assert len(self.relationship_dict) == len(Member._RELATION_SCALES), "The number of relationship matrices and the number of relationship matrix scales are inconsistent"

        # Record actions (output and clear every Island._RECORD_PERIOD)
        self.record_action_dict = {
            "attack": {},
            "benefit": {},
            "benefit_land": {},
        }
        self.record_born = []
        self.record_death = []

        # Record state (append a 0 to the end every Island._RECORD_PERIOD)
        self.record_total_production = [0]
        self.record_total_consumption = [0]
        self.record_total_dict = {
            "attack": [0],
            "benefit": [0],
            "benefit_land": [0],
        }
        self.record_historic_ratio_list = np.array([(0,0,0,0)])
        self.record_historic_ranking_list = [(0,0,0)]
        self.record_payoff_matrix = np.zeros((8, 64))
        self.record_land = [self.land.owner_id]
        
        self.previous_vitalities = {}
        self.vitality_diff = {}

        # Round number
        self.current_round = 0


    ############################################################################
    ################################ Basic Operations #################################### 

    def member_by_name(
        self,
        name: str,
    ) -> Member:
        for member in self.current_members:
            if member.name == name:
                return member
        
        for member in self.all_members:
            if member.name == name:
                return member

        raise KeyError(f"Member {name} not found!")

    # =============================== Member Addition and Removal ===================================
    def _backup_member_list(
        self, 
        member_list: List[Member]
    ) -> List[Member]:
        """Copy member_list"""
        return [member for member in member_list]

    def _member_list_append(
        self, 
        append: List[Member] = [], 
        appended_rela_rows: np.ndarray = [], 
        appended_rela_columns: np.ndarray = [],
    ) -> None:
        """
        Add a list of members to current_members and all_members,
        Increase current_member_num,
        Modify relationships matrix,
        Modify member surviver_id,
        """
        appended_num = len(append)
        prev_member_num = self.current_member_num

        if not isinstance(appended_rela_columns, np.ndarray):
            raise ValueError("The added columns of the relationship matrix should be of ndarray type")
        if not isinstance(appended_rela_rows, np.ndarray):
            raise ValueError("The added rows of the relationship matrix should be of ndarray type")
        assert appended_rela_columns.shape == (prev_member_num, appended_num), "Input relationship column shape does not match"
        assert appended_rela_rows.shape == (appended_num, prev_member_num), "Input relationship row shape does not match"

        # Add members to the list
        for member in append:
            member.surviver_id = self.current_member_num
            self.current_members.append(member)
            self.all_members.append(member)

            self.current_member_num += 1

        # Record births
        self.record_born = self.record_born + append

        # Modify relationship matrix
        for key in self.relationship_dict.keys():
            # Cannot directly assign, need to modify the original array size and fill in values
            tmp_old = self.relationship_dict[key].copy()
            tmp_new = np.zeros((self.current_member_num, self.current_member_num))
            
            tmp_new[:prev_member_num, :prev_member_num] = tmp_old
            tmp_new[:prev_member_num, prev_member_num:] = appended_rela_columns
            tmp_new[prev_member_num:, :prev_member_num] = appended_rela_rows
            np.fill_diagonal(tmp_new, np.nan)

            self.relationship_dict[key] = tmp_new

        return

    def _member_list_drop(
        self, 
        drop: List[Member] = []
    ) -> None:
        """
        Remove members from current_members,
        Decrease current_member_num,
        Modify relationships matrix,
        Reassign all members' surviver_id
        """
        drop_id = np.array([member.id for member in drop])            # Check id to ensure correct deletion
        drop_sur_id = np.array([member.surviver_id for member in drop])

        if (drop_sur_id == None).any():
            raise AttributeError(f"The object to be deleted should have a surviver_id")

        for member in drop:
            assert member.owned_land == [], "The object to be deleted should not have land"

        # Sort to ensure correct removal
        argsort_sur_id = np.argsort(drop_sur_id)[::-1]
        drop_id = drop_id[argsort_sur_id]
        drop_sur_id = drop_sur_id[argsort_sur_id]
        
        # Remove members from the list
        for idx in range(len(drop_id)):
            id_to_drop = drop_id[idx]
            sur_id_to_drop = drop_sur_id[idx]
            assert self.current_members[sur_id_to_drop].id == id_to_drop, "Deleted object does not match"

            self.current_members[sur_id_to_drop].surviver_id = None

            del self.current_members[sur_id_to_drop]
            self.current_member_num -= 1

        # Modify relationship matrix
        for key in self.relationship_dict.keys():
            # Cannot directly assign, need to modify the original array size and fill in values
            tmp = np.delete(self.relationship_dict[key], drop_sur_id, axis=0)
            tmp = np.delete(tmp, drop_sur_id, axis=1)

            self.relationship_dict[key] = tmp

        # Reassign surviving members
        for sur_id in range(self.current_member_num):
            self.current_members[sur_id].surviver_id = sur_id
        
        return

    def member_list_modify(
        self, 
        append: List[Member] = [], 
        drop: List[Member] = [], 
        appended_rela_rows: np.ndarray = np.empty(0), 
        appended_rela_columns: np.ndarray = np.empty(0)
    ) -> None:
        """
        Modify member_list, first add members, then modify
        Record births and deaths
        """
        if append != []:
            self._member_list_append(
                append=append, 
                appended_rela_rows=appended_rela_rows, appended_rela_columns=appended_rela_columns
            )
        if drop != []:
            self._member_list_drop(
                drop=drop
            )

        return

    @property
    def is_dead(self,) -> bool:
        return self.current_member_num == 0

# ================================ Relationship Matrix Modification ==================================
    def _overlap_of_relations(
        self, 
        principal: Member, 
        object: Member,
        normalize: bool = True,
        ) -> List[float]:
        """Calculate the inner product of the relationship network"""

        def _normalize(arr):
            """Remove nan, normalize vector"""
            arr[principal.surviver_id] = 0
            arr[object.surviver_id] = 0
            norm = np.linalg.norm(arr)
            if norm == 0:
                return 0
            else:
                if normalize:
                    return arr / norm
                else:
                    return arr

        overlaps = []
        for relationship in list(self.relationship_dict.values()):
            
            pri_row = _normalize(relationship[principal.surviver_id, :].copy())
            pri_col = _normalize(relationship[:, principal.surviver_id].copy())
            obj_row = _normalize(relationship[object.surviver_id, :].copy())
            obj_col = _normalize(relationship[:, object.surviver_id].copy())

            overlaps.append((
                np.sum(np.sqrt(pri_row * obj_row))
                + np.sum(np.sqrt(pri_row * obj_col))
                + np.sum(np.sqrt(pri_col * obj_row))
                + np.sum(np.sqrt(pri_col * obj_col))) / 4
            )
        
        return overlaps

    def _relations_w_normalize(
        self,
        principal: Member,
        object: Member,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calculate the normalized (tanh) relationship matrix element"""
        elements = []
        for relationship in list(self.relationship_dict.values()):
            elements.append(relationship[principal.surviver_id, object.surviver_id])
            elements.append(relationship[object.surviver_id, principal.surviver_id])

        elements = np.array(elements)
        
        if normalize:
            result = elements * np.repeat(Member._RELATION_SCALES, 2)
            return np.tanh(result)
        else:
            return elements

    def relationship_modify(
        self, 
        relationship_name,
        member_1: Member, 
        member_2: Member, 
        add_value: float
    ) -> None:
        """
        Increase matrix element [member_1.surviver_id, member_2.surviver_id]
        """
        assert member_1 is not member_2, "Cannot modify diagonal elements in the relationship matrix"
        relationship = self.relationship_dict[relationship_name]
        relationship[member_1.surviver_id, member_2.surviver_id] += add_value

# =================================== Land ======================================
    def _acquire_land(
        self, 
        member: Member, 
        location: Tuple[int, int],
    ) -> None:
        assert self.land[location] is None, "The acquired land should have no owner"

        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = member
        member.acquire_land(location)

    def _discard_land(
        self,
        member: Member,
        location: Tuple[int, int],
    ) -> None:
        assert location in member.owned_land, "Can only discard owned land"
        assert self.land[location] == member, "Can only discard one's own land"

        loc_0, loc_1 = location
        self.land.owner[loc_0][loc_1] = None
        member.discard_land(location)

    def get_neighbors(self):
        for member in self.current_members:
            self._get_neighbors(member, backend="inner product")

    def _get_neighbors(
        self, 
        member: Member,
        backend: Optional[str] = None,
    ) -> None:
        """
        Store four lists:
        - clear_list: allows passage
        - self_blocked_list: directly adjacent to member
        - neighbor_blocked_list: members indirectly adjacent to member and the landlords acting as bridges, stored in the format (landlord, indirectly adjacent member)
        - empty_loc_list: idle land
        """
        (
            member.current_clear_list,
            member.current_self_blocked_list,
            member.current_neighbor_blocked_list,
            member.current_empty_loc_list,
        ) = self.land.neighbors(
            member, 
            self, 
            Island._NEIGHBOR_SEARCH_RANGE,
            # decision_threshold=1,
            backend,
        )

    def _maintain_neighbor_list(self, member: Member):
        """
        Remove already deceased neighbors
        """
        member.current_clear_list = [
            neighbor for neighbor in member.current_clear_list 
            if not self.all_members[neighbor].autopsy()
        ]
        member.current_self_blocked_list = [
            neighbor for neighbor in member.current_self_blocked_list 
            if not self.all_members[neighbor].autopsy()
        ]
        member.current_neighbor_blocked_list = [
            neighbor for neighbor in member.current_neighbor_blocked_list 
            if not self.all_members[neighbor[0]].autopsy() and not self.all_members[neighbor[1]].autopsy()
        ]
        member.current_empty_loc_list = [
            loc for loc in member.current_empty_loc_list
            if self.land[loc] is None
        ]

    def _find_targets(
        self,
        member: Member,
        target_list: List[Member],
        decision_name: str,
        prob_of_action: float = 1.0,
        other_requirements: Callable = None,
        bilateral: bool = False,
        land_owner_decision: str = "",
        backend: Optional[str] = None,
    ) -> List[Member]:
        """
        Select objects from the potential object list based on the decision function
        decision_name: one of the keys of Member.parameter_dict
        other_requirements: function, input two Members, output True (passed requirement)/False
        bilateral: set to True for the decision function to be mutually compliant
        landlord_decision: landlord's decision
        """

        if target_list == []:
            return []

        selected_target = []
        for tgt in target_list:

            if self._rng.uniform(0, 1) > prob_of_action:
                continue

            # Check if tgt is a tuple (including landlord) or a single member
            if isinstance(tgt, tuple):
                land_owner, obj = tgt
            elif isinstance(tgt, Member):
                obj = tgt
                land_owner = None
            else:
                raise ValueError("Please input the correct target in the list: member, or (landlord, member)")

            # Check if obj has been judged repeatedly
            if obj in selected_target:
                continue

            if other_requirements is not None:
                if not other_requirements(member, obj):
                    continue

            if not member.decision(
                decision_name,
                obj,
                self,
                backend=backend,
                logger=self._logger,
            ):  
                continue

            if bilateral:
                if not obj.decision(
                    decision_name,
                    member,
                    self,
                    backend=backend,
                    logger=self._logger,
                ):
                    continue

            if land_owner_decision != "" and land_owner is not None:
                if not land_owner.decision(
                    land_owner_decision,
                    obj,
                    self,
                    backend=backend,
                ):
                    continue

            selected_target.append(obj)

        return list(set(selected_target))

# ##############################################################################
# ##################################### Record ####################################
    def _record_actions(
        self, 
        record_name: str, 
        member_1: Member, 
        member_2: Member, 
        value_1: float, 
        value_2: float = None
    ):
        record_dict = self.record_action_dict[record_name]

        # Record actions of both parties
        try:
            record_dict[(member_1.id, member_2.id)] += value_1
        except KeyError:
            record_dict[(member_1.id, member_2.id)] = value_1
        if value_2 is not None:
            try:
                record_dict[(member_2.id, member_1.id)] += value_2
            except KeyError:
                record_dict[(member_2.id, member_1.id)] = value_2

        # Record total actions
        if value_2 is not None:
            self.record_total_dict[record_name][-1] += value_1 + value_2
        else:
            self.record_total_dict[record_name][-1] += value_1
    
    def generate_decision_history(self) -> None:
        if not hasattr(self, 'decision_history'):
            self.decision_history = {}
            
        for member_1 in self.all_members:
            if member_1.id not in self.decision_history:
                self.decision_history[member_1.id] = {}
            self.decision_history[member_1.id][self.current_round] = (0, 0, 0)
        
        for (member_1, member_2) in self.record_action_dict['attack']: 
            ## member_1 here is member_1.id
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (1, prev_decisions[1], prev_decisions[2])
        
        for (member_1, member_2) in self.record_action_dict['benefit']:
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (prev_decisions[0], 1, prev_decisions[2])
        
        for (member_1, member_2) in self.record_action_dict['benefit_land']:
            prev_decisions = self.decision_history[member_1][self.current_round]
            self.decision_history[member_1][self.current_round] = (prev_decisions[0], prev_decisions[1], 1)

    def record_historic_ratio(self):
        current_attack_ratio = self.record_total_dict['attack'][-1]/(self.current_member_num)
        current_benefit_ratio = self.record_total_dict['benefit'][-1]/(self.current_member_num)
        current_benefit_land_ratio = self.record_total_dict['benefit_land'][-1]/(self.current_member_num)
        # current_reproduce_ratio = self.record_total_dict[-1]['reproduce']/(self.current_member_num) #Reserved for reproduce

        self.record_historic_ratio_list = np.append(self.record_historic_ratio_list, [[current_attack_ratio, current_benefit_ratio, current_benefit_land_ratio, 0]], axis=0)

    def record_historic_ranking(self):
        # Calculate ranking
        current_attack_ranking = (sorted(self.record_historic_ratio_list[:,0]).index(self.record_historic_ratio_list[:,0][-1]) + 1)/len(self.record_historic_ratio_list[:,0])
        current_benefit_ranking = (sorted(self.record_historic_ratio_list[:,1]).index(self.record_historic_ratio_list[:,1][-1]) + 1)/len(self.record_historic_ratio_list[:,1])
        current_benefit_land_ranking = (sorted(self.record_historic_ratio_list[:,2]).index(self.record_historic_ratio_list[:,2][-1]) + 1)/len(self.record_historic_ratio_list[:,2])
        self.record_historic_ranking_list.append((current_attack_ranking, current_benefit_ranking, current_benefit_land_ranking))

    def calculate_histoic_quartile(self):
        # Flatten the list of tuples
        flattened_list = [value for t in self.record_historic_ranking_list for value in t]

        # Sort the flattened list
        sorted_list = sorted(flattened_list)

        # Determine quartile boundaries
        q1_boundary = sorted_list[len(sorted_list) // 4]
        q2_boundary = sorted_list[len(sorted_list) // 2]
        q3_boundary = sorted_list[3 * len(sorted_list) // 4]

        def determine_quartile(value):
            if value <= q1_boundary:
                return 1
            elif value <= q2_boundary:
                return 2
            elif value <= q3_boundary:
                return 3
            else:
                return 4

        # Map each value in the tuples to its corresponding quartile
        # Use a dictionary with the round as the key and the quartile tuple as the value
        self.record_historic_quartile_dict = {current_round: (determine_quartile(val1), determine_quartile(val2), determine_quartile(val3)) 
                                            for current_round, (val1, val2, val3) in enumerate(self.record_historic_ranking_list)}

    def generate_collective_actions_transition_matrix(self):
        # Importing the required libraries and retrying the process

        # Given list of tuples (sequence of events)
        events = self.record_historic_quartile_dict

        # Generate all possible tuple states
        all_states = [(i, j, k) for i in range(1, 5) for j in range(1, 5) for k in range(1, 5)]

        # Create a dictionary to track transitions
        transitions = defaultdict(lambda: defaultdict(int))

        # Populate the transitions dictionary
        for i in range(len(events) - 1):
            current_state = events[i]
            next_state = events[i + 1]
            transitions[current_state][next_state] += 1

        # Normalize the counts to get probabilities
        for current_state, next_states in transitions.items():
            total_transitions = sum(next_states.values())
            for next_state, count in next_states.items():
                transitions[current_state][next_state] = count / total_transitions

        # Create the 64x64 transition matrix
        transition_matrix = []

        for current_state in all_states:
            row = []
            for next_state in all_states:
                row.append(transitions[current_state].get(next_state, 0))
            transition_matrix.append(row)

        self.collective_actions_transition_matrix = transition_matrix
    
    def compute_vitality_difference(self):
        round_diff = {}
        for member in self.current_members:
            current_vitality = member.vitality
            prev_vitality = self.previous_vitalities.get(member, current_vitality)  # Default to current vitality if not found
            round_diff[member] = current_vitality - prev_vitality
            self.previous_vitalities[member] = current_vitality
        self.vitality_diff[self.current_round] = round_diff

    def compute_payoff_matrix(self):
        action_combinations = [(i, j, k) for i in [0,1] for j in [0,1] for k in [0,1]]
        tuple_states = list(self.record_historic_quartile_dict.values())
        payoff_matrix = np.zeros((8, 64))
        quartile_combinations = list(itertools.product(range(1, 5), repeat=3))

        for idx_a, action_a in enumerate(action_combinations):
            for idx_t, tuple_state in enumerate(tuple_states):
                combination_index = quartile_combinations.index(tuple_state)
                total_vitality_change = 0
                count = 0

                # Assuming each member has a decision history
                for member in self.current_members:
                    try:
                        decisions = self.decision_history[member]  # Access the decision history from the dictionary in the Island class
                        for round, decision in decisions.items():
                            if decision == action_a and tuple_state == self.record_historic_quartile_dict[round]:
                                if round > 4:
                                    total_vitality_change += self.vitality_diff[round-1][member] + self.vitality_diff[round-2][member] + self.vitality_diff[round-3][member]
                                count += 1
                    except KeyError:
                        pass # for newborn babies

                if count != 0:
                    avg_vitality_change = total_vitality_change / count
                else:
                    avg_vitality_change = 0


                payoff_matrix[idx_a][combination_index] = avg_vitality_change

                self.record_payoff_matrix = payoff_matrix
    
    ############################################################################
    def save_current_island(self, path):
        current_member_df = self.current_members[0].save_to_row()
        for sur_id in range(1, self.current_member_num):
            current_member_df = pd.concat([
                current_member_df,
                self.current_members[sur_id].save_to_row()],
                axis=0
            )
        
        info_df = pd.DataFrame({
            "_create_from_file": [self._create_from_file],
            "_file_name": [self._file_name],
            "_seed": [self._random_seed],
            "init_member_num": [self.init_member_num],
            "current_member_num": [self.current_member_num],
            "current_round": [self.current_round],
            "land_shape": [f"{self.land.shape[0]} {self.land.shape[1]}"],
        })

        relationship_df = pd.DataFrame()
        for key, rela in self.relationship_dict.items():
            rela_df = pd.DataFrame(rela, index=None, columns=None)
            relationship_df = pd.concat([relationship_df, rela_df], axis=0)

        # Actions saved before this round
        action_list = []
        for action_name, action_dict in self.record_action_dict.items():
            sub_action_info = [[key[0], key[1], value] for key, value in action_dict.items()]
            sub_action_df = pd.DataFrame(
                sub_action_info,
                columns=[f"{action_name}_1", f"{action_name}_2", "value"]
            )
            action_list.append(sub_action_df)
        born_df = pd.DataFrame(
            [member.id for member in self.record_born], 
            columns=["born"]
        )
        action_list.append(born_df)
        death_df = pd.DataFrame(
            [member.id for member in self.record_death], 
            columns=["death"]
        )
        action_list.append(death_df)
        action_df = pd.concat(action_list, axis=1)

        # Land
        land_df = pd.DataFrame(
            self.land.owner_id()
        )
        
        current_member_df.to_csv(path + "members.csv")
        info_df.to_csv(path + "island_info.csv")
        relationship_df.to_csv(path + "relationships.csv")
        action_df.to_csv(path + "action.csv")
        land_df.to_csv(path + "land.csv")

    @classmethod
    def load_island(cls, path):

        current_member_df = pd.read_csv(path + "members.csv")
        info_df = pd.read_csv(path + "island_info.csv")
        relationship_df = pd.read_csv(path + "relationships.csv")
        action_df = pd.read_csv(path + "action.csv")
        land_df = pd.read_csv(path + "land.csv")

        """
        Not finished yet
        """

    def save_to_pickle(self, file_name: str) -> None:

        sys.setrecursionlimit(500000)

        file = open(file_name, 'wb') 
        pickle.dump(self, file)

    @classmethod
    def load_from_pickle(cls, file_name: str) -> "Island":
        file = open(file_name, 'rb') 
        return pickle.load(file)

    ############################################################################
    ################################## Simulation #####################################
    @property 
    def shuffled_members(self) -> List[Member]:
        """
        Shuffle the entire current_members list
        """
        shuffled_members = self._backup_member_list(self.current_members)
        self._rng.shuffle(shuffled_members)

        return shuffled_members

    def declare_dead(self, member: Member):
        # Immediately lose all land
        for loc in member.owned_land.copy():
            self._discard_land(member, loc)
        # Remove member_2
        self.member_list_modify(drop=[member])
        # Record death
        self.record_death.append(member)
        # log
        self._logger.info(f"{member.name} has died.")

    def produce(self) -> None:
        """
        Production  

            1. Increase food storage based on productivity and land
        """
        for member in self.current_members:
            self.record_total_production[-1] += member.produce()

    def _attack(
        self, 
        member_1: Member, 
        member_2: Member
    ) -> None:
        # Calculate attack and steal values
        strength_1 = member_1.strength
        steal_1 = member_1.steal

        if steal_1 > member_2.cargo:
            steal_1 = member_2.cargo

        # Settle attack and steal
        member_2.vitality -= strength_1
        member_2.cargo -= steal_1

        # Modify relationship matrix
        self.relationship_modify("victim", member_2, member_1, strength_1 + steal_1)

        # Record action
        self._record_actions(
            "attack",
            member_1,
            member_2,
            strength_1 + steal_1,
        )

        # Settle death
        if member_2.autopsy():
            # Settle death
            self.declare_dead(member_2)
            # member_1 immediately gets an expansion opportunity
            self._expand(member_1)

        # If the color of the attack target is the same as its own, the attacker restores color (exits the organization)
        if np.allclose(member_1._current_color, member_2._current_color):
            member_1._current_color = member_1._color.copy()


    def fight(
        self, 
        prob_to_fight: float = 1.0
        ):
        """
        Fight
        """
        for member in self.shuffled_members:
            if member.autopsy():
                # If the member died in a previous round, no action will be taken
                continue

            self._maintain_neighbor_list(member)
            
            # Find targets from neighbors
            target_list = self._convert_id_list_to_member_list(
                member.current_clear_list 
                + member.current_self_blocked_list 
                + member.current_neighbor_blocked_list
            )
            attack_list = self._find_targets(
                member = member,
                target_list = target_list,
                decision_name = "attack",
                prob_of_action = prob_to_fight,
                other_requirements = None,
                bilateral = False,
                land_owner_decision = "attack"
            )

            for target in attack_list:
                self._attack(member, target)

    def _offer(
        self, 
        member_1: Member, 
        member_2: Member,
        parameter_influence: bool = True
    ) -> None:
        """
        member_1 gives to member_2
        If member_1 can give less than 1, it will not give
        """
        amount = member_1.offer

        if amount < 1:
            return 

        # Settle giving
        member_2.cargo += amount
        member_1.cargo -= amount

        # Modify relationship matrix
        self.relationship_modify("benefit", member_2, member_1, amount)

        # Record
        if amount > 1e-15:
            self._record_actions(
                "benefit",
                member_1,
                member_2,
                amount,
                None
            )

        # The parameters of the recipient are affected
        if parameter_influence:
            member_2.parameter_absorb(
                [member_2, member_1],
                [1 - Member._PARAMETER_INFLUENCE, Member._PARAMETER_INFLUENCE],
                0
            )

        # The recipient is colored
        member_2._current_color = member_1._current_color

    def _convert_id_list_to_member_list(self, id_list: List[int | Tuple[int,int]]) -> List[Member]:
        """
        Convert id_list to member_list
        """
        member_list = []
        for id in id_list:
            if isinstance(id, tuple):
                land_owner = self.all_members[id[0]]
                obj = self.all_members[id[1]]
                member_list.append((land_owner, obj))
            else:
                member_list.append(self.all_members[id])

        return member_list
        
    def trade(
        self,
        prob_to_trade: float = 1.0
        ):
        """
        Trade and communication
        """
        for member in self.shuffled_members:            
            self._maintain_neighbor_list(member)
            
            # Find targets from neighbors
            trade_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(
                    member.current_clear_list 
                    + member.current_self_blocked_list 
                    + member.current_neighbor_blocked_list
                ),
                decision_name = "offer",
                prob_of_action = prob_to_trade,
                other_requirements = _requirement_for_offer,
                bilateral = False,
                land_owner_decision = "offer"
            )

            self._rng.shuffle(trade_list)
            for target in trade_list:
                self._offer(member, target, parameter_influence=True)

    def _expand(
        self,
        member: Member,
    ):
        """
        Expand
        """
        self._maintain_neighbor_list(member)
        if len(member.current_empty_loc_list) > 0:
            self._acquire_land(member, member.current_empty_loc_list[0])

    def colonize(
        self,
    ) -> None:
        """
        Collective expansion
        """
        for member in self.shuffled_members:
            self._expand(member)

    def consume(
        self, 
    ):
        """
        Consumption

            1. Calculate consumption. Consumption will gradually increase with age 
            2. Deduct consumption from vitality, if vitality is less than zero, it is recorded as death
            3. Eat food from the warehouse to restore vitality
            4. If there are death cases, update the collective list, update the numbering, update the relationship matrix
        """
        for member in self.current_members:
            consumption = member.consume()

            # Record
            self.record_total_consumption[-1] += consumption

            if member.autopsy():
                self.declare_dead(member)

        for member in self.current_members:
            member.recover()

    def _offer_land(
        self, 
        member_1: Member, 
        member_2: Member,
        parameter_influence: bool = True,
        assigned_pos: float = None,
    ) -> None:
        """
        member_1 gives to member_2.  
        Select the land that is farthest from oneself and closest to the other party.  
        When providing an "ideal" position, it will automatically select the land closest to assigned_pos from the giver's land.
        """
        
        # Select the land that is farthest from oneself and closest to the other party
        if member_1.land_num == 0:
            raise RuntimeError("The person giving land should own at least one piece of land")
        if member_1.land_num == 0 and assigned_pos is None:
            raise RuntimeError("In the absence of a specified position, the person receiving land should own at least one piece of land")

        pos_1 = member_1.center_of_land(self.land)
        if assigned_pos is None:
            pos_2 = member_2.center_of_land(self.land)
        else:
            pos_2 = assigned_pos

        farthest_distance = -np.inf
        for land in member_1.owned_land:
            distance = self.land.distance(pos_1, land) - self.land.distance(pos_2, land)
            # distance = np.sum((pos_1 - land)**2) - np.sum((pos_2 - land)**2)
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_pos = land
        pos = farthest_pos

        # Settle giving
        self._discard_land(member_1, pos)
        self._acquire_land(member_2, pos)

        # Modify relationship matrix
        self.relationship_modify("benefit_land", member_2, member_1, 1)

        # Record
        self._record_actions(
            "benefit_land",
            member_1,
            member_2,
            1,
            None
        )

        # The parameters of the recipient are affected
        if parameter_influence:
            member_2.parameter_absorb(
                [member_2, member_1],
                [1 - Member._PARAMETER_INFLUENCE, Member._PARAMETER_INFLUENCE],
                0
            )
            
        # The recipient is colored
        member_2._current_color = member_1._current_color

    def land_distribute(
        self,
        prob_to_distr: float = 1.0
        ):
        """
        Trade and communication
        """
        for member in self.shuffled_members:
            self._maintain_neighbor_list(member)
            
            # Find targets from neighbors
            distr_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(
                    member.current_clear_list 
                    + member.current_self_blocked_list 
                    + member.current_neighbor_blocked_list
                ),
                decision_name = "offer_land",
                prob_of_action = prob_to_distr,
                other_requirements = _requirement_for_offer_land,
                bilateral = False,
                land_owner_decision = "offer_land"
            )

            distr_len = len(distr_list)
            if distr_len > 0:
                target_idx = self._rng.choice(
                    range(distr_len),
                    size = 1
                )
                target = distr_list[int(target_idx)]
                self._offer_land(member, target, parameter_influence=True)

    def _bear(
        self,
        member_1: Member, 
        member_2: Member,
    ):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = Member.born(
            member_1,
            member_2,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        self.member_list_modify(
            append=[child],
            appended_rela_columns=np.zeros((self.current_member_num, 1)),
            appended_rela_rows=np.zeros((1, self.current_member_num)),
        )
        
        # Calculate the initial vitality given to the child by the parents
        vitality_base = Member._CHILD_VITALITY
        # Total resources of both parties
        member_1_total = member_1.cargo + member_1.vitality
        member_2_total = member_2.cargo + member_2.vitality
        total = member_1_total + member_2_total
        # Deduct losses proportionally
        member_1_give = vitality_base * member_1_total / total
        member_2_give = vitality_base * member_2_total / total
        if member_1.cargo >= member_1_give:
            member_1.cargo -= member_1_give
        else:
            member_1.vitality -= (member_1_give - member_1.cargo)
            member_1.cargo = 0
        if member_2.cargo >= member_2_give:
            member_2.cargo -= member_2_give
        else:
            member_2.vitality -= (member_2_give - member_2.cargo)
            member_2.cargo = 0
        child.vitality = vitality_base
        # The child remembers the contributions of the parents
        self.relationship_modify("benefit", child, member_1, member_1_give)
        self.relationship_modify("benefit", child, member_2, member_2_give)

        # Parents unconditionally offer the child once
        self._offer(member_1, child, parameter_influence=False)
        self._offer(member_2, child, parameter_influence=False)

        # Parents unconditionally give land to the child
        center_pos = (member_1.center_of_land(self.land) + member_2.center_of_land(self.land)) / 2
        for _ in range(Member._LAND_HERITAGE):
            self._offer_land(
                member_1, child, parameter_influence=False, 
                assigned_pos=center_pos)
            self._offer_land(
                member_2, child, parameter_influence=False, 
                assigned_pos=center_pos)

        # The child calculates and recovers vitality
        child.recover()

        self._logger.info(f"\t{member_1} and {member_2} gave birth to {child}")

    def reproduce(
        self, 
        prob_of_reproduce: float = 1.0
    ):
        """
        Reproduction

            1. Select individuals who meet the age criteria
            2. Randomly group, sort within the group.
            3. Iterate within each group, based on the [reproduction decision] function, determine mutual affection, select parents
            4. Check if both parties meet the reproduction conditions (the sum of vitality and cargo)
            5. Parents deduct cargo proportionally, the total is a fixed value, and if the cargo is insufficient, vitality is deducted.
            6. Produce a child. Set the child's age (0), parents. The child randomly inherits the basic attributes and decision parameters of the parents, adding **a little** random fluctuation. The child's initial vitality is a fixed value (less than the parents' consumption), stored...
            7. Parents unconditionally offer the child once
            8. If there are birth cases, update the collective list, update numbering, update relationship matrix
            
        """

        for member in self.shuffled_members:
            self._maintain_neighbor_list(member)
            
            # Find targets from neighbors
            partner_list = self._find_targets(
                member = member,
                target_list = self._convert_id_list_to_member_list(member.current_clear_list),
                decision_name = "reproduce",
                prob_of_action = prob_of_reproduce,
                other_requirements = _requirement_for_reproduction,
                bilateral = True,
                land_owner_decision = ""
            )

            distr_len = len(partner_list)
            if distr_len > 0:
                target_idx = self._rng.choice(
                    range(distr_len),
                    size = 1
                )
                target = partner_list[int(target_idx)]
                self._bear(member, target)

    def _record_init_per_period(
        self,
    ):
        for key in self.record_action_dict.keys():
            self.record_action_dict[key] = {}
        for key in self.record_total_dict.keys():
            self.record_total_dict[key].append(0)

        self.record_total_consumption.append(0)
        self.record_total_production.append(0)

        self.record_born = []
        self.record_death = []

    def new_round(self, save_file: bool = True, log_status=False):
        # Output content
        if self.current_round % Island._RECORD_PERIOD == 0:
            # Save
            if save_file:
                self.save_to_pickle(self._save_path + f"{self.current_round:d}.island")

            # Output
            if log_status:
                self.log_status(lang=self.log_lang)

            # Initialize storage
            self._record_init_per_period()

        # Round number +1
        self.current_round += 1

        # Each surviving member ages by one year
        for member in self.current_members:
            member.age += 1

    def log_status(
        self,
        action = False,
        summary = True,
        members = True,
        log_instead_of_print = True,
        ):
        
        lang=self.log_lang  # Added language parameter for multilingual support
        
        log_str = ""

        log_str += "#" * 21 + f" {self.current_round:d} " + "#" * 21 + "\n"

        if action:
            log_str += "=" * 21 + self._translate("attack", lang) + "=" * 21 + "\n"
            if self.record_action_dict["attack"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["attack"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

            log_str += "=" * 21 + self._translate("benefit", lang) + "=" * 21 + "\n"
            if self.record_action_dict["benefit"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n" 

            log_str += "=" * 20 + self._translate("benefit_land", lang) + "=" * 20 + "\n"
            if self.record_action_dict["benefit_land"] != {}:
                for (mem_1, mem_2), value in self.record_action_dict["benefit_land"].items():
                    member_1 = self.all_members[mem_1]
                    member_2 = self.all_members[mem_2]
                    log_str += f"\t{member_1} --{value:.1f}-> {member_2} \n"

        if summary:
            log_str += "=" * 50 + "\n"
            log_str += f"{self._translate('born_this_round', lang)}: {self.record_born} \n"
            log_str += f"{self._translate('died_this_round', lang)}: {self.record_death} \n"
            log_str += f"{self._translate('total_benefit_this_round', lang)}: {self.record_total_dict['benefit'][-1]:.1f} \n"
            log_str += f"{self._translate('total_attack_this_round', lang)}: {self.record_total_dict['attack'][-1]:.1f} \n"
            log_str += f"{self._translate('total_production_this_round', lang)}: {self.record_total_production[-1]:.1f} \n"
            log_str += f"{self._translate('total_consumption_this_round', lang)}: {self.record_total_consumption[-1]:.1f} \n"
            log_str += f"{self._translate('active_ratio_this_round', lang)}: {self.record_historic_ratio_list[-1]} \n"
            log_str += f"{self._translate('historical_ranking_this_round', lang)}: {self.record_historic_ranking_list[-1]} \n"
            
        if members:
            log_str += "=" * 50 + "\n"
            log_str += self._translate("member_header", lang)  # Use translation for the header
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

                if not log_instead_of_print:
                    mem_str = colored(
                            member._current_color,
                            mem_str
                        )
            
                log_str += mem_str
            
        if log_instead_of_print:
            self._logger.info(log_str)
            return log_str
        else:
            print(log_str)
            return log_str

    def _translate(self, key: str, lang: str) -> str:
        translations = {
            'en': {
                'attack': 'Attack',
                'benefit': 'Benefit',
                'benefit_land': 'Land Benefit',
                'born_this_round': 'Born this round',
                'died_this_round': 'Died this round',
                'total_benefit_this_round': 'Total Benefit this round',
                'total_attack_this_round': 'Total Attack this round',
                'total_production_this_round': 'Total Production this round',
                'total_consumption_this_round': 'Total Consumption this round',
                'active_ratio_this_round': 'Active Ratio this round',
                'historical_ranking_this_round': 'Historical Ranking this round',
                'member_header': '\t ID Sur_ID  Name          Age   Vitality    Cargo    Land Count \n',  # Updated translation for member header
            },
            'cn': {
                'attack': '攻击',
                'benefit': '给予',
                'benefit_land': '给予土地',
                'born_this_round': '本轮出生',
                'died_this_round': '本轮死亡',
                'total_benefit_this_round': '本轮总给予',
                'total_attack_this_round': '本轮总攻击',
                'total_production_this_round': '本轮总产量',
                'total_consumption_this_round': '本轮总消耗',
                'active_ratio_this_round': '本轮活跃比率',
                'historical_ranking_this_round': '本轮比率历史排位',
                'member_header': '\t ID Sur_ID  姓名          年龄   血量    仓库    土地数 \n',  # Added translation for member header
            },
            'jp': {
                'attack': '攻撃',
                'benefit': '利益',
                'benefit_land': '土地の利益',
                'born_this_round': '今ラウンドで生まれた',
                'died_this_round': '今ラウンドで死亡',
                'total_benefit_this_round': '今ラウンドの総利益',
                'total_attack_this_round': '今ラウンドの総攻撃',
                'total_production_this_round': '今ラウンドの総生産',
                'total_consumption_this_round': '今ラウンドの総消費',
                'active_ratio_this_round': '今ラウンドのアクティブ比率',
                'historical_ranking_this_round': '今ラウンドの歴史的ランキング',
                'member_header': '\t ID Sur_ID  名前          年齢   生命力    貨物    土地数 \n',  # Added translation for member header
            },
            'sp': {
                'attack': 'Ataque',
                'benefit': 'Beneficio',
                'benefit_land': 'Beneficio de Tierra',
                'born_this_round': 'Nacidos esta ronda',
                'died_this_round': 'Muertos esta ronda',
                'total_benefit_this_round': 'Beneficio Total esta ronda',
                'total_attack_this_round': 'Ataque Total esta ronda',
                'total_production_this_round': 'Producción Total esta ronda',
                'total_consumption_this_round': 'Consumo Total esta ronda',
                'active_ratio_this_round': 'Ratio Activo esta ronda',
                'historical_ranking_this_round': 'Ranking Histórico esta ronda',
                'member_header': '\t ID Sur_ID  Nombre          Edad   Vitalidad    Carga    Conteo de Tierra \n',  # Updated translation for member header
            }
        }
        
        return translations.get(lang, translations[self.log_lang]).get(key, key)
        
    def record_statistics(self):
        """
        Statistics
        """
        if self.current_member_num > 0:
            self.record_historic_ratio()
            self.record_historic_ranking()
            self.calculate_histoic_quartile()
            self.generate_collective_actions_transition_matrix()
            self.generate_decision_history()
            self.compute_vitality_difference()
            self.compute_payoff_matrix()