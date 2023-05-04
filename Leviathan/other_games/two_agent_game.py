import numpy as np

from Leviathan.Island import IslandBase, MemberBase

from multiprocessing import Pool

from typing import List

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

def Hawk_Dove_pay_off(resource: float, conflict_loss: float):
    # Hawk Dove game
    # it's returns the pay off for both of the players
    half_res = resource / 2
    half_loss = conflict_loss / 2
    matrix = np.array([
        [half_res - half_loss, resource],
        [0, half_res]
    ])

    return matrix

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
        child.strategy = parent.strategy

        return child

    def mutation(self):
        self.strategy = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]


class HDAgent(MemberBase):
    def __init__(
        self,
        name, 
        id,
        surviver_id,
        rng,
        decision_num = 3,
        mixed_strategy = False,
    ):
        """
        An agent plays strategy with probability unconditioned on the history.
        I gave this class the name is because I initially want to simulate the
        Hawk Dove game, but it can be used for other games as well.
        """
        super().__init__(name, id, surviver_id, rng)

        self.fitness = 0

        self.decision_num = decision_num
        self.mixed_strategy = mixed_strategy
        self.strategy = rng.dirichlet(np.ones(self.decision_num), size=1)[0]

        self.parent = None
        self.ancestor = self
        

class Game(IslandBase):
    def __init__(
        self,
        init_member_num,
        pay_off_matrix,
        mixed_strategy = False,
        random_seed = None
    ):
        super().__init__(init_member_num, random_seed)

        self.init_member_num = init_member_num
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
    
    def bear_new_member(self, parent):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = SBAgent.born(
            parent,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )

        self.member_list_modify(append = [child])

    def discard_member_by_payoff(self, discard_rate):
        # discard the worst member
        discard_num = int(len(self.current_members) * discard_rate)
        if discard_num == 0:
            return
        discarded_member = self.sort_member()[-discard_num:]

        self.member_list_modify(drop = discarded_member)

    def bear_new_member_by_payoff(self):
        bare_num = self.init_member_num - len(self.current_members)
        bare_member = self.sort_member()[:bare_num]

        for parent in bare_member:
            self.bear_new_member(parent)

    def mutation(self, mutation_rate):
        # mutation_rate: the probability of mutation
        for member in self.current_members:
            if self._rng.uniform() < mutation_rate:
                member.mutation()
    
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

        -1 in each history matrix represent no history.
        """
    
        super().__init__(init_member_num, pay_off_matrix, mixed_strategy, random_seed)

        self.history_len = history_len
        for idx in range(-1, -1-self.history_len, -1):
            self.relationship_dict[f"history {idx:d}"] = (
                np.ones((self.init_member_num, self.init_member_num), dtype=int) * -1
            )       

    def _insert_history(
        self, 
        surviver_id_1, surviver_id_2,
        decision_1, decision_2,
    ):
        # insert the history of the game between surviver_id_1 and surviver_id_2
        # if self.relationship_dict["history_1"][surviver_id_1, surviver_id_2] is not -1,
        # then the history_1[surviver_id_1, surviver_id_2] will be replaced by the new game
        # and the old game will be moved to history_2[surviver_id_1, surviver_id_2]

        if self.history_len == 0:
            return

        for idx in range(-self.history_len+1, 0, 1):
            self.relationship_dict[f"history {idx-1:d}"][surviver_id_1, surviver_id_2] = (
                self.relationship_dict[f"history {idx:d}"][surviver_id_1, surviver_id_2]
            )

        self.relationship_dict["history -1"][surviver_id_1, surviver_id_2] = decision_1
        self.relationship_dict["history -1"][surviver_id_2, surviver_id_1] = decision_2

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

    def relationship_network(self, )
        