import numpy as np

from Leviathan.Island import IslandBase, MemberBase

import copy
from multiprocessing import Pool

from typing import List, Callable, Tuple

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
        self.strategy = rng.dirichlet(np.ones(self.decision_num), size=1)[0]

        self.parent = None
        self.ancestor = self

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
        MemberBase.__init__(self, name, id, surviver_id, rng)

        self.fitness = 0

        self.decision_num = decision_num
        self.mixed_strategy = mixed_strategy

        if isinstance(history_keys, int | float):
            self.history_keys = self._history_keys_by_length(history_keys)
        else:
            self.history_keys = history_keys
        self.decision_num = decision_num
        self.strategy = dict([
            (key, rng.dirichlet(np.ones(self.decision_num), size=1)[0]) for key in self.history_keys
        ])

        self.parent = None
        self.ancestor = self

    def _history_keys_by_length(self, length):
        # the last key should be (np.nan, np.nan), but it doesn't work
        # because nan != nan in python. Should manually convert between
        # np.nan and None
        keys_1_history = [*np.ndindex((self.decision_num, self.decision_num))] + [(None, None)]
        
        if length == 0:
            return [tuple()]
        
        key_list = []
        for ndidx in np.ndindex((self.decision_num**2 + 1, ) * length):
            key = []
            for i in ndidx:
                key.append(keys_1_history[i])
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
        for key in self.history_keys:
            if self._rng.uniform() < mutation_rate:
                self.strategy[key] = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]

    def strategy_drift(self, drift_rate, drift_strength):
        for key in self.history_keys:
            self.strategy[key] = self._single_strategy_drift(
                self.strategy[key], drift_rate, drift_strength
            )


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

        self.member_list_modify(append = [child])

    def discard_randomly(self, discard_rate):
        discarded_member = []
        for member in self.current_members:
            if self._rng.uniform() < discard_rate:
                discarded_member.append(member)

        self.member_list_modify(drop = discarded_member)

    def discard_member_by_payoff(self, discard_rate):
        # discard the worst member
        discard_num = int(len(self.current_members) * discard_rate)
        if discard_num == 0:
            return
        discarded_member = self.sort_member()[-discard_num:]

        self.member_list_modify(drop = discarded_member)

    def bear_new_member_by_payoff(self, strategy_drift_rate=1, strategy_drift_strength=0.05):
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


class GameWHistoryS1Tn(GameWHistory):
    """
    S1T1: when using history information for decision, history with spacial distance 1 and 
    any temporal distance will be considered
    """

    def __init__(
        self, 
        init_member_num,
        pay_off_matrix, 
        decision_history_len: int = 1,
        history_len: int = 1, 
        mixed_strategy=False,
        random_seed=None
    ):
        IslandBase.__init__(self, init_member_num, random_seed)

        assert decision_history_len <= history_len, "decision_history_len should be smaller than history_len"
        self.decision_history_len = decision_history_len

        self.init_members = [HDAgent(
            self._NAME_LIST[i], 
            id=i, 
            surviver_id=i, 
            rng=self._rng,
            history_keys=decision_history_len,
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
        
        idx = (player_1.surviver_id, player_2.surviver_id)

        history_key_for_player_1 = []
        history_key_for_player_2 = []
        
        for his_idx in range(-self.decision_history_len, 0, 1):
            strategy_1 = self.relationship_dict[f"history {his_idx:d}"][idx]
            strategy_2 = self.relationship_dict[f"history {his_idx:d}"][idx[::-1]]

            if np.isnan(strategy_1):
                strategy_1 = None
            if np.isnan(strategy_2):
                strategy_2 = None

            if strategy_1 is None:
                assert strategy_2 is None, f"The history is not symmetric between player {player_1.name} and {player_2.name}."
            if strategy_2 is None:
                assert strategy_1 is None, f"The history is not symmetric between player {player_1.name} and {player_2.name}."

            history_key_for_player_1.append(
                (strategy_1, strategy_2)
            )
            history_key_for_player_2.append(
                (strategy_2, strategy_1)
            )

        return tuple(history_key_for_player_1), tuple(history_key_for_player_2)

    def single_game(self, player_1: HDAgent, player_2: HDAgent):
        # inputting self into the decision function is for future use, in case some 
        # decision function need to know the history of the game     

        his_1, his_2 = self.history_key_by_player(player_1, player_2)

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

