import numpy as np

from Leviathan.Island import IslandBase, MemberBase

import copy
from multiprocessing import Pool

from typing import List, Callable, Tuple

# ##############################################################################

def suicide_bomber_pay_off(benefit_kill, cost):
    # follow Adami et al.
    matrix = np.array([
        [1, 1, 0],
        [1-cost, 1-cost, 1-cost],
        [1-2*cost+benefit_kill, 1-2*cost, 1-2*cost]
    ])

    return matrix

def RPS_pay_off(win: float, lose: float):
    # Rock-Paper-Scissors game
    matrix = np.array([
        [0.0, -lose, win],
        [win, 0.0, -lose],
        [-lose, win, 0.0]
    ])

    return matrix

def Hawk_Dove_pay_off(resource: float, conflict_loss: float, off_set: float):
    # Hawk Dove game
    # it's returns the pay off for both of the players
    half_res = resource / 2
    half_loss = conflict_loss / 2
    matrix = np.array([
        [half_res - half_loss, resource],
        [0, half_res]
    ])

    return matrix + off_set

# ##############################################################################

def _HD_intermediate_history_reduction(intermed_history_chain):
    # reduce a key looks like ((0, 1), (1, 0), ...) to (0, 1)
    # key: (A_to_B, B_to_A)
    # 0: Hawk, 1: Dove
    # rule: A Hawk B + B Hawk C = A Dove C
    #       A Hawk B + B Dove C = A Hawk C
    #       A Dove B + B Dove C = A Dove C
    #       A Dove B + B Hawk C = A Hawk C
    # or in other words, if there is a even number of Hawks in the chain, then the result is Dove

    reduced_elem_1, reduced_elem_2 = 0, 0
    for elem in intermed_history_chain:
        elem_1, elem_2 = elem
        if elem_1 is None:
            return None, None
        if elem_2 is None:
            return None, None

        reduced_elem_1 += elem_1
        reduced_elem_2 += elem_2

    reduced_elem_1 %= 2
    reduced_elem_2 %= 2

    return reduced_elem_1, reduced_elem_2

def _HD_majority_vote(key_list ):
    # key in key_list should be a tuple of length 2, representing the history of two agents

    # count the number of (0, 0), (0, 1), (1, 0), (1, 1), (None, None)
    # use the dictionary to keep the order of the keys, it's benifitial when there is a tie
    # in the majority vote, so the first key will be chosen
    count = {}
    for key in key_list:
        if key not in count.keys():
            count[key] = 1
        else:
            count[key] += 1
    
    # sort the count by the value
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    
    # if there is a tie, then the first key will be chosen
    return sorted_count[0][0]

def HD_spacial_key_simplifier_1(original_key):
    # for spacial keys, reduce the intermidiate interaction
    his_key, spacial_key = original_key

    new_spacial_key = []
    for key in spacial_key:
        # key is a spacial key between a list of agents with length spacial_depth
        # reduce the intermidiate interaction
        reduced_key = _HD_intermediate_history_reduction(key)
        new_spacial_key.append(reduced_key)

    return his_key, tuple(new_spacial_key)

def HD_spacial_key_simplifier_2(original_key):
    # for spacial keys, reduce the intermidiate interaction & majority vote

    his_key, spacial_key = HD_spacial_key_simplifier_1(original_key)

    new_spacial_key = _HD_majority_vote(spacial_key)

    return his_key, new_spacial_key


# ##############################################################################
class SBAgent(MemberBase):
    def __init__(
        self,
        name, 
        id,
        surviver_id,
        rng,
        decision_num,
        mixed_strategy = False,
    ):
        """
        An agent plays strategy with probability unconditioned on the history.
        I gave this class the name is because I initially want to simulate the
        Suicide Bomber game, but it can be used for other games as well.
        """
        super().__init__(name, id, surviver_id, rng)

        self.fitness = 0

        self.decision_num = decision_num
        self.mixed_strategy = mixed_strategy

        self.parent = None
        self.ancestor = self

        self._strategy_init()

    def _strategy_init(self):
        self.strategy = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]

    def decision(self, *args, **kwargs):
        if self.mixed_strategy:
            return self._rng.choice(self.decision_num, p=self.strategy)
        else:
            return np.argmax(self.strategy)
    
    def __repr__(self) -> str:
        return f"{self.name}: fitness({self.fitness}), strategy({self.strategy})"

    @classmethod
    def born(
        cls, 
        parent: "SBAgent", 
        name: str,
        id: int,
        surviver_id: int,
        rng: np.random.Generator,
    ) -> "SBAgent":
        child = SBAgent(
            name, id, surviver_id, rng, 
            parent.decision_num, mixed_strategy=parent.mixed_strategy,
        )

        child.parent = parent
        child.ancestor = parent.ancestor
        child.strategy = parent.strategy.copy()

        return child

    def mutation(self, mutation_rate):
        if self._rng.uniform() < mutation_rate:
            self.strategy = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]

    def _single_strategy_drift(self, strategy, drift_rate, drift_strength):
        """
        randomly apply a gaussian noise to the strategy with mean 0 and std drift_strength
        """
        if drift_rate == 0:
            return strategy

        if self._rng.uniform() < drift_rate:
            strategy += self._rng.normal(0, drift_strength, size=self.decision_num)
            strategy = np.clip(strategy, 0, None)
            strategy /= np.sum(strategy)

        return strategy
    
    def strategy_drift(self, drift_rate, drift_strength):
        self.strategy = self._single_strategy_drift(self.strategy, drift_rate, drift_strength)


class HDAgent(SBAgent):
    def __init__(
        self,
        name, 
        id,
        surviver_id,
        rng,
        history_keys: List | int = 0,
        decision_num = 3,
        mixed_strategy = False,
    ):
        """
        An agent plays strategy with probability unconditioned on the history.

        history_keys can be a list of keys or an integer indicating the history_length. 
        For history_length = 0, it's the same as SBAgent. For history_length > 0, the 
        probability to use a strategy becomes conditioned on the history of the game. There 
        are in total (decision_num ** 2 + 1) ** history_length conditions. Those conditional
        probabilities are stored in a dictionary. As an example, for history_length = 3,
        key can be ((1, 2), (np.nan, np.nan), (0, 0)), where np.nan represents no interaction

        I gave this class the name is because I initially want to simulate the
        Hawk Dove game, but it can be used for other games as well.
        """
        self.history_keys = history_keys

        super().__init__(  
            name, id, surviver_id, rng, decision_num, mixed_strategy
        )

    def _strategy_init(self):
        if isinstance(self.history_keys, int | float):
            self.history_keys = self._strategy_keys_by_history_length(self.history_keys)

        self.strategy_keys = self.history_keys

        self.decision_num = self.decision_num
        self.strategy = dict([
            (key, self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]) for key in self.strategy_keys
        ])

    def _strategy_keys_by_history_length(self, length):
        # the last key should be (np.nan, np.nan), but it doesn't work
        # because nan != nan in python. Should manually convert between
        # np.nan and None

        # a list of keys for a single history
        keys_1_history = [*np.ndindex((self.decision_num, self.decision_num))] + [(None, None)]
        
        if length == 0:
            return [tuple()]
        
        # combine the keys for multiple history, so in the end it looks like
        # ((1, 2), (np.nan, np.nan), (0, 0), ...)
        key_list = self._key_tensor_key(*([keys_1_history] * length))

        return key_list

    @staticmethod
    def _key_tensor_key(*args):
        """
        args should be (key_list_1, key_list_2, ...) with length (l1, l2, ...)

        Return (l1*l2*...) keys, each key is a sum of the keys in the corresponding
        position of the input key_list
        """

        length_list = [len(key_list) for key_list in args]

        key_list = []
        for ndidx in np.ndindex(tuple(length_list)):
            key = []
            for dim_i, idx in enumerate(ndidx):
                key.append(args[dim_i][idx])
            key_list.append(tuple(key))

        return key_list

    def decision(self, history_key):
        strategy = self.strategy[history_key]
        if self.mixed_strategy:
            return self._rng.choice(self.decision_num, p=strategy)
        else:
            return np.argmax(strategy)
        
    @classmethod
    def born(
        cls, 
        parent: "HDAgent", 
        name: str,
        id: int,
        surviver_id: int,
        rng: np.random.Generator,
    ) -> "HDAgent":
        child = HDAgent(
            name, id, surviver_id, rng, 
            parent.history_keys, parent.decision_num, mixed_strategy=parent.mixed_strategy,
        )

        child.parent = parent
        child.ancestor = parent.ancestor
        child.strategy = copy.deepcopy(parent.strategy)

        return child

    def mutation(self, mutation_rate):
        for key in self.strategy_keys:
            if self._rng.uniform() < mutation_rate:
                self.strategy[key] = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]

    def strategy_drift(self, drift_rate, drift_strength):
        for key in self.strategy_keys:
            self.strategy[key] = self._single_strategy_drift(
                self.strategy[key], drift_rate, drift_strength
            )


class LeviathanAgent(HDAgent):
    def __init__(
        self,
        name, 
        id,
        surviver_id,
        rng,
        spacial_keys: List | int = 1,
        spacial_depth: int = 1,
        apply_chain_rule: bool = False,
        history_keys: List | int = 0,
        mixed_strategy = False,
    ):
        """
        An agent plays strategy with probability unconditioned on the history between whom
        he's competing with & their connected agents' history. For this class only, 
        decision_num has to be 2, representing Hawk and Dove. 

        Parameters
        ----------
        history_keys:
            history_keys can be a list of keys or an integer indicating the history_length. 
            For history_length = 0, it's the same as SBAgent. For history_length > 0, the 
            probability to use a strategy becomes conditioned on the history of the game. There 
            are in total (decision_num ** 2 + 1) ** history_length conditions. Those conditional
            probabilities are stored in a dictionary. As an example, for history_length = 3,
            key can be ((1, 2), (np.nan, np.nan), (0, 0)), where np.nan represents no interaction
        spatial_keys, spatial_depth:
            spacial_keys can be a list of keys or an integer indicating the how many possible
            (indirect) interaction path are considered. spacial_depth indicates how many agents
            are considered in each path. For example, if spacial_depth = 2, then interaction with
            one agent in between is considered. In the end, each of the spacial key (if turned to an 
            array) has shape (spacial_keys, spacial_depth, 2)

            agent's strategy is then a combination of history_keys and spacial_keys. 
        apply_chain_rule:
            If true, then the spacial key will be reduced by the chain rule. For example, if
            spacial_depth = 2, then the spacial history key ((0, 1), (0, 0)) will be 
            reduced to (0, 1)

        """
        self.spacial_keys = spacial_keys
        self.spacial_depth = spacial_depth
        self.apply_chain_rule = apply_chain_rule

        super().__init__(
            name, id, surviver_id, rng, history_keys, 2, mixed_strategy
        )

    def _strategy_init(self):
        if isinstance(self.history_keys, int | float):
            self.history_keys = self._strategy_keys_by_history_length(self.history_keys)
            
        if isinstance(self.spacial_keys, int | float):
            self.spacial_keys = self._strategy_keys_by_spacial_length(self.spacial_keys)

        self.strategy_keys = self._key_tensor_key(self.history_keys, self.spacial_keys)

        self.strategy = dict([
            (key, self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]) for key in self.strategy_keys
        ])

    def _strategy_keys_by_spacial_length(self, length):
        # a list of keys for a single history. It looks like ((1, 2), (np.nan, np.nan))
        # because it contains the history of two agents and their intermedate agents
        
        if length == 0:
            return [tuple()]
        
        if self.apply_chain_rule:
            one_key = self._strategy_keys_by_history_length(1)
            one_key = [key[0] for key in one_key]
        else:
            one_key = self._strategy_keys_by_history_length(self.spacial_depth)

        # combine the keys for multiple history, so in the end it looks like
        # ((1, 2), (np.nan, np.nan), (0, 0), ...)
        key_list = self._key_tensor_key(*([one_key] * length))

        return key_list

    @classmethod
    def born(
        cls, 
        parent: "LeviathanAgent", 
        name: str,
        id: int,
        surviver_id: int,
        rng: np.random.Generator,
    ) -> "LeviathanAgent":
        child = LeviathanAgent(
            name, id, surviver_id, rng, 
            parent.spacial_keys, spacial_depth=parent.spacial_depth, apply_chain_rule=parent.apply_chain_rule,
            history_keys=parent.history_keys, 
            mixed_strategy=parent.mixed_strategy,
        )

        child.parent = parent
        child.ancestor = parent.ancestor
        child.strategy = copy.deepcopy(parent.strategy)

        return child


# ##############################################################################
# ##############################################################################
class Game(IslandBase):
    def __init__(
        self,
        init_member_num,
        pay_off_matrix,
        mixed_strategy = False,
        random_seed = None
    ):
        super().__init__(init_member_num, random_seed)

        self.init_members = [SBAgent(
            self._NAME_LIST[i], 
            id=i, 
            surviver_id=i, 
            rng=self._rng,
            decision_num=pay_off_matrix.shape[0],
            mixed_strategy=mixed_strategy,
        ) for i in range(self.init_member_num)]
        self.all_members: List[SBAgent] = self._backup_member_list(self.init_members)
        self.current_members: List[SBAgent] = self._backup_member_list(self.init_members)

        self.payoff_matrix = pay_off_matrix

    def single_game(self, player_1: SBAgent, player_2: SBAgent):   
        # inputting self into the decision function is for future use, in case some 
        # decision function need to know the history of the game     
        player_1_strategy = player_1.decision(self)
        player_2_strategy = player_2.decision(self)

        if len(self.payoff_matrix.shape) == 2:
            # a symmetric game (A terminology invented by me)
            fit_1 = self.payoff_matrix[player_1_strategy, player_2_strategy]
            fit_2 = self.payoff_matrix[player_2_strategy, player_1_strategy]
        elif len(self.payoff_matrix.shape) == 3:
            # an asymmetric game
            fit_1, fit_2 = self.payoff_matrix[player_1_strategy, player_2_strategy]
        else:
            raise ValueError("The shape of the payoff matrix is not correct.")

        # I should update the player's fitness here, but I want to test the multi-processing
        # so I will update the fitness in the multiple_game function.
        # However, it turn out that the multi-processing is very slow...
        return fit_1, fit_2

    def multiple_game(self, game_num, cpu_num=1):
        # use for loop
        if cpu_num == 1:
            for _ in range(game_num):
                mem_1, mem_2 = self.random_pick(2)
                fit_1, fit_2 = self.single_game(mem_1, mem_2)
                mem_1.fitness += fit_1
                mem_2.fitness += fit_2
            return
        
        # use multiprocessing.Pool, very slow
        game_member_list = []
        for _ in range(game_num):
            game_member_list.append(self.random_pick(2))

        with Pool(cpu_num) as p: 
            result = p.starmap(
                self.single_game, 
                game_member_list,
            )

        # update fitness
        for mem_1, mem_2 in game_member_list:
            fit_1, fit_2 = result.pop(0)
            mem_1.fitness += fit_1
            mem_2.fitness += fit_2

    def random_pick(self, pick_num):
        # pick_num: the number of member to pick
        return self._rng.choice(self.current_members, size=pick_num, replace=False)

    def sort_member(self):
        # sort by fitness (best comes first) and return a new list
        # but keep the order in the self.current_members
        sorted_member = sorted(self.current_members, key=lambda x: x.fitness, reverse=True)
        return sorted_member
    
    def bear_new_member(self, parent, strategy_drift_rate=1.0, strategy_drift_strength=0.05):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = SBAgent.born(
            parent,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        child.strategy_drift(strategy_drift_rate, strategy_drift_strength)

        self.member_list_modify(append = [child])

    def discard_randomly(self, discard_rate):
        discarded_member = []
        for member in self.current_members:
            if self._rng.uniform() < discard_rate:
                discarded_member.append(member)

        self.member_list_modify(drop = discarded_member)

    def bear_randomly(self, strategy_drift_rate=1.0, strategy_drift_strength=0.05):
        bare_num = self.init_member_num - len(self.current_members)
        bare_member = []
        for _ in range(bare_num):
            parent = self._rng.choice(self.current_members)
            bare_member.append(parent)

        for parent in bare_member:
            self.bear_new_member(parent, strategy_drift_rate, strategy_drift_strength)

    def discard_member_by_payoff(self, discard_rate):
        # discard the worst member
        discard_num = int(len(self.current_members) * discard_rate)
        if discard_num == 0:
            return
        discarded_member = self.sort_member()[-discard_num:]

        self.member_list_modify(drop = discarded_member)

    def bear_new_member_by_payoff(self, strategy_drift_rate=1.0, strategy_drift_strength=0.05):
        bare_num = self.init_member_num - len(self.current_members)
        bare_member = self.sort_member()[:bare_num]

        for parent in bare_member:
            self.bear_new_member(parent, strategy_drift_rate, strategy_drift_strength)

    def mutation(self, mutation_rate):
        # mutation_rate: the probability of mutation
        for member in self.current_members:
            member.mutation(mutation_rate)
    
    def reset_fitness(self):
        for member in self.current_members:
            member.fitness = 0

class GameWHistory(Game):
    def __init__(
        self,
        init_member_num,
        pay_off_matrix,
        history_len: int = 1,
        mixed_strategy = False,
        random_seed = None
    ):
        """
        history's idx runs from -1 to -history_len, indicating how close the history is 
        in the past.
        
        the history{idx}[id_1, id_2] works as a queue for 
        storing the history of between two players with survival_id id_1 and id_2. 
        the last game is at the top of the queue. 
        the history{idx}[id_1, id_2] records the strategy of id_1 done to id_2. 

        nan in each history matrix represent no history.
        """
    
        super().__init__(init_member_num, pay_off_matrix, mixed_strategy, random_seed)

        self.history_len = history_len
        empty_history = np.full((self.init_member_num, self.init_member_num), np.nan, dtype=float)
        for idx in range(-1, -1-self.history_len, -1):
            self.relationship_dict[f"history {idx:d}"] = (
                empty_history.copy()
            )       

    def _insert_history(
        self, 
        surviver_id_1, surviver_id_2,
        decision_1, decision_2,
    ):
        # insert the history of the game between surviver_id_1 and surviver_id_2
        # if self.relationship_dict["history_1"][surviver_id_1, surviver_id_2] is not nan,
        # then the history_1[surviver_id_1, surviver_id_2] will be replaced by the new game
        # and the old game will be moved to history_2[surviver_id_1, surviver_id_2]

        if self.history_len == 0:
            return
        
        if not (
            self.relationship_dict["history -1"][surviver_id_1, surviver_id_2] == np.nan
            and self.relationship_dict["history -1"][surviver_id_2, surviver_id_1] == np.nan
        ):
            # there are two reason that the slot is empty: either they really have no history
            # or the history is too old that it has been moved to the next history slot
            for idx in range(-self.history_len+1, 0, 1):
                self.relationship_dict[f"history {idx-1:d}"][surviver_id_1, surviver_id_2] = (
                    self.relationship_dict[f"history {idx:d}"][surviver_id_1, surviver_id_2]
                )
                self.relationship_dict[f"history {idx-1:d}"][surviver_id_2, surviver_id_1] = (
                    self.relationship_dict[f"history {idx:d}"][surviver_id_2, surviver_id_1])

        self.relationship_dict["history -1"][surviver_id_1, surviver_id_2] = decision_1
        self.relationship_dict["history -1"][surviver_id_2, surviver_id_1] = decision_2

    def forget(self):
        # forget the oldest history and move all the history one slot forward
        if self.history_len == 0:
            return
        
        for idx in range(-self.history_len+1, 0, 1):
            self.relationship_dict[f"history {idx-1:d}"] = (
                self.relationship_dict[f"history {idx:d}"].copy()
            )

        self.relationship_dict[f"history -1"] = (
            np.full((self.init_member_num, self.init_member_num), np.nan, dtype=float)
        )

    def single_game(self, player_1: SBAgent, player_2: SBAgent):
        # inputting self into the decision function is for future use, in case some 
        # decision function need to know the history of the game     
        player_1_strategy = player_1.decision(self)
        player_2_strategy = player_2.decision(self)

        if len(self.payoff_matrix.shape) == 2:
            # a symmetric game (A terminology invented by me)
            fit_1 = self.payoff_matrix[player_1_strategy, player_2_strategy]
            fit_2 = self.payoff_matrix[player_2_strategy, player_1_strategy]
        elif len(self.payoff_matrix.shape) == 3:
            # an asymmetric game
            fit_1, fit_2 = self.payoff_matrix[player_1_strategy, player_2_strategy]
        else:
            raise ValueError("The shape of the payoff matrix is not correct.")

        # update history
        self._insert_history(
            player_1.surviver_id, player_2.surviver_id,
            player_1_strategy, player_2_strategy,
        )

        # I should update the player's fitness here, but I want to test the multi-processing
        # so I will update the fitness in the multiple_game function.
        # However, it turn out that the multi-processing is very slow...
        return fit_1, fit_2

    def multiple_game(self, game_num):
        return super().multiple_game(game_num, cpu_num=1)
    
    def bear_new_member(self, parent, strategy_drift_rate, strategy_drift_strength):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = SBAgent.born(
            parent,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        child.strategy_drift(strategy_drift_rate, strategy_drift_strength)

        # empty history
        empty_history_row = np.full((1, self.current_member_num), np.nan)
        empty_history_column = np.full((self.current_member_num, 1), np.nan)

        self.member_list_modify(
            append = [child],
            appended_rela_columns=empty_history_column,
            appended_rela_rows=empty_history_row,
        )

    def majority_voting(self, strategy_list):
        """
        strategy_list should be like [0, 1, 1, nan, ...]

        Reuturn the most common strategy in this list. If there is a tie, then
        a random key that ties will be chosen
        """
        # convert nan to decision_num
        strategy_list = np.array(strategy_list)
        strategy_list[np.isnan(strategy_list)] = len(self.payoff_matrix)
        strategy_list = strategy_list.astype(int)

        # count
        count_list = np.bincount(strategy_list)

        # find the max count
        max_count = np.max(count_list)
        largest_strategies = np.where(count_list == max_count)[0]
        largest_strategy = self._rng.choice(largest_strategies)

        # replace decision_num with nan
        if largest_strategy == len(self.payoff_matrix):
            largest_strategy = np.nan

        return largest_strategy


    def majority_voting_pairs(self, strategy_list):
        """
        strategy_list should be like [(0, 1), (1, 0), (1, 0), (nan, nan), ...]

        Reuturn the most common strategy pair in this list. If there is a tie, then
        a random key that ties will be chosen
        """
        his_12, his_21 = zip(*strategy_list)

        # majority voting, find the most common strategy 0, 1 or nan for each player
        largest_strategy_12 = self.majority_voting(his_12)
        largest_strategy_21 = self.majority_voting(his_21)

        if np.isnan(largest_strategy_12) or np.isnan(largest_strategy_21):
            return (np.nan, np.nan)

        return (largest_strategy_12, largest_strategy_21)
    
    def history_by_player(
        self, player_1: SBAgent, player_2: SBAgent, 
        history_len: int = 2,
        majority_voted: bool = False,
    ):
        """
        Return the history of the game between player_1 and player_2. It'll look like 
        [(0, 1), (1, 0), (1, 0), (nan, nan), ...] and [(1, 0), (0, 1), (0, 1), (nan, nan), ...)]. 
        Both of the list has length history_len.
        
        If majority_voted is True, then the history will be majority voted and return a single pair. 
        """
        if history_len > self.history_len:
            raise ValueError("history_len should be smaller than self.history_len")
        
        his_1 = []
        his_2 = []
        for idx in range(-1, -history_len-1, -1):
            his_1.append(self.relationship_dict[f"history {idx:d}"][player_1.surviver_id, player_2.surviver_id])
            his_2.append(self.relationship_dict[f"history {idx:d}"][player_2.surviver_id, player_1.surviver_id])

        history_12 = (zip(his_1, his_2))
        history_21 = (zip(his_2, his_1))

        if majority_voted:
            history_12 = self.majority_voting_pairs(history_12)
            history_21 = history_12[::-1]
            return (history_12,), (history_21,)
        else:
            return tuple(history_12), tuple(history_21)



class GameWHistoryS1Tn(GameWHistory):
    """
    S1Tn: when using history information for decision, history with spacial distance 1 and 
    any temporal distance n will be considered. 
    """

    def __init__(
        self, 
        init_member_num,
        pay_off_matrix, 
        decision_history_len: int = 1,
        history_len: int = 1,
        apply_majority_voting: bool = False, 
        mixed_strategy=False,
        random_seed=None
    ):
        IslandBase.__init__(self, init_member_num, random_seed)

        assert decision_history_len <= history_len, "decision_history_len should be smaller than history_len"
        self.decision_history_len = decision_history_len

        if apply_majority_voting:
            agent_decision_history_len = 1
        else:
            agent_decision_history_len = decision_history_len
        self.apply_majority_voting = apply_majority_voting

        self.init_members = [HDAgent(
            self._NAME_LIST[i], 
            id=i, 
            surviver_id=i, 
            rng=self._rng,
            history_keys=agent_decision_history_len,
            decision_num=pay_off_matrix.shape[0],
            mixed_strategy=mixed_strategy,
        ) for i in range(self.init_member_num)]
        self.all_members: List[HDAgent] = self._backup_member_list(self.init_members)
        self.current_members: List[HDAgent] = self._backup_member_list(self.init_members)

        self.payoff_matrix = pay_off_matrix

        self.history_len = history_len
        empty_history = np.full((self.init_member_num, self.init_member_num), np.nan, dtype=float)
        for idx in range(-1, -1-self.history_len, -1):
            self.relationship_dict[f"history {idx:d}"] = (
                empty_history.copy()
            )     

    def history_key_by_player(self, player_1: HDAgent, player_2: HDAgent):
        # should convert nan to None, because nan != nan in python. It's a
        # bad idea to use (nan, nan) as dictionary keys
        history_12, history_21 = self.history_by_player(
            player_1, player_2,
            history_len=self.decision_history_len,
            majority_voted=self.apply_majority_voting,
        )
        history_12 = [
            (None, None) if np.isnan(key[0]) else key for key in history_12
        ]
        history_21 = [
            (None, None) if np.isnan(key[0]) else key for key in history_21
        ]

        return tuple(history_12), tuple(history_21)
        

    def strategy_key_by_player(self, player_1: HDAgent, player_2: HDAgent):
        return self.history_key_by_player(player_1, player_2)

    def single_game(self, player_1: HDAgent, player_2: HDAgent):
        # inputting self into the decision function is for future use, in case some 
        # decision function need to know the history of the game     

        his_1, his_2 = self.strategy_key_by_player(player_1, player_2)

        player_1_strategy = player_1.decision(his_1)
        player_2_strategy = player_2.decision(his_2)

        if len(self.payoff_matrix.shape) == 2:
            # a symmetric game (A terminology invented by me)
            fit_1 = self.payoff_matrix[player_1_strategy, player_2_strategy]
            fit_2 = self.payoff_matrix[player_2_strategy, player_1_strategy]
        elif len(self.payoff_matrix.shape) == 3:
            # an asymmetric game
            fit_1, fit_2 = self.payoff_matrix[player_1_strategy, player_2_strategy]
        else:
            raise ValueError("The shape of the payoff matrix is not correct.")

        # update history
        self._insert_history(
            player_1.surviver_id, player_2.surviver_id,
            player_1_strategy, player_2_strategy,
        )

        # I should update the player's fitness here, but I want to test the multi-processing
        # so I will update the fitness in the multiple_game function.
        # However, it turn out that the multi-processing is very slow...
        return fit_1, fit_2 
    
    def bear_new_member(self, parent, strategy_drift_rate, strategy_drift_strength):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = HDAgent.born(
            parent,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        child.strategy_drift(strategy_drift_rate, strategy_drift_strength)

        # empty history
        empty_history_row = np.full((1, self.current_member_num), np.nan)
        empty_history_column = np.full((self.current_member_num, 1), np.nan)

        self.member_list_modify(
            append = [child],
            appended_rela_columns=empty_history_column,
            appended_rela_rows=empty_history_row,
        )

class GameWHistorySnD1Tn(GameWHistoryS1Tn):
    """
    SnD1Tn: when using history information for decision, history with spacial number n, 
    spacial distance 1, and temporal distance n will be considered.

    Note that spacial number isn't spacial distance(depth). It's the number of agents
    that has relationship with both of the players. 
    """

    def __init__(
        self, 
        init_member_num,
        pay_off_matrix, 
        decision_spacial_len: int = 1,
        decision_history_len: int = 1,
        apply_chain_rule: bool = True,
        apply_majority_voting: bool = True,
        history_len: int = 1, 
        mixed_strategy=False,
        random_seed=None
    ):
        """
        Parameters
        ----------
        apply_chain_rule:
            If true, then the spacial key will be reduced by the chain rule. For example, if
            the spacial history key is ((0, 1), (0, 0)), it will be
            reduced to (0, 1)
        apply_majority_voting:
            If true, then the spacial key will be reduced by the majority voting. For example, if
            spacial_len
        """
        IslandBase.__init__(self, init_member_num, random_seed)

        if not apply_majority_voting:
            raise NotImplementedError("apply_majority_voting = False is not implemented yet.")
        if not apply_chain_rule:
            raise NotImplementedError("apply_chain_rule = False is not implemented yet.")
        if decision_spacial_len != init_member_num - 2:
            raise NotImplementedError("decision_spacial_len != init_member_num - 2 is not implemented yet.")
        

        assert decision_history_len <= history_len, "decision_history_len should be smaller than history_len"
        self.decision_history_len = decision_history_len
        self.decision_spacial_len = decision_spacial_len
        self.apply_chain_rule = apply_chain_rule
        self.apply_majority_voting = apply_majority_voting

        if apply_majority_voting:
            agent_decision_spacial_len = 1
            agent_decision_history_len = 1
        else:
            agent_decision_spacial_len = decision_spacial_len
            agent_decision_history_len = decision_history_len

        self.init_members = [LeviathanAgent(
            name=self._NAME_LIST[i], 
            id=i, 
            surviver_id=i, 
            rng=self._rng,
            spacial_keys=agent_decision_spacial_len,
            spacial_depth=1,
            apply_chain_rule=apply_chain_rule,
            history_keys=agent_decision_history_len,
            mixed_strategy=mixed_strategy,
        ) for i in range(self.init_member_num)]
        self.all_members: List[LeviathanAgent] = self._backup_member_list(self.init_members)
        self.current_members: List[LeviathanAgent] = self._backup_member_list(self.init_members)

        self.payoff_matrix = pay_off_matrix
        
        self.history_len = history_len
        empty_history = np.full((self.init_member_num, self.init_member_num), np.nan, dtype=float)
        for idx in range(-1, -1-self.history_len, -1):
            self.relationship_dict[f"history {idx:d}"] = (
                empty_history.copy()
            )     

    def spacial_key_by_player(self, player_1: LeviathanAgent, player_2: LeviathanAgent):
        # find the spacial key for the two players
        # find "self.spacial_keys" players that has relationship with player_1 and player_2
        # stored in self.relationship_dict["history -1"]
        
        # look for players that has relationship with both player_1 and player_2
        
        if not self.apply_majority_voting:
            raise NotImplementedError("apply_majority_voting = False is not implemented yet.")
        if not self.apply_chain_rule:
            raise NotImplementedError("apply_chain_rule = False is not implemented yet.")
        if self.decision_spacial_len != self.init_member_num - 2:
            raise NotImplementedError("decision_spacial_len != init_member_num - 2 is not implemented yet.")
        
        idx_1, idx_2 = player_1.surviver_id, player_2.surviver_id

        his_copy = self.relationship_dict["history -1"].copy()
        his_copy[idx_1, idx_2] = np.nan
        his_copy[idx_2, idx_1] = np.nan

        # using chain rule to reduce the length of spacial key
        # rule: Hawk + Hawk = Dove, Hawk + Dove = Dove, Dove + Dove = Dove
        his_12 = (his_copy[idx_1, :] + his_copy[:, idx_2]) % 2
        his_21 = (his_copy[idx_2, :] + his_copy[:, idx_1]) % 2
        decision_pair_list = np.array([his_12, his_21]).T

        # majority voting, find the most common strategy 0, 1 or nan for each player
        major_history = self.majority_voting_pairs(decision_pair_list)

        # replace nan with None
        if np.isnan(major_history[0]):
            major_history = (None, None)

        return (major_history,), (major_history[::-1],)

    def strategy_key_by_player(self, player_1: LeviathanAgent, player_2: LeviathanAgent):
        his_1, his_2 = self.history_key_by_player(player_1, player_2)

        spacial_1, spacial_2 = self.spacial_key_by_player(player_1, player_2)

        return (his_1, spacial_1), (his_2, spacial_2)
    
    def bear_new_member(self, parent, strategy_drift_rate, strategy_drift_strength):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = LeviathanAgent.born(
            parent,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng,
        )
        child.strategy_drift(strategy_drift_rate, strategy_drift_strength)

        # empty history
        empty_history_row = np.full((1, self.current_member_num), np.nan)
        empty_history_column = np.full((self.current_member_num, 1), np.nan)

        self.member_list_modify(
            append = [child],
            appended_rela_columns=empty_history_column,
            appended_rela_rows=empty_history_row,
        )