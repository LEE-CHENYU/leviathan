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

def Hawk_Dove_pay_off(resource: float, conflict_lose: float):
    # Hawk Dove game
    # it's not a single 
    matrix = np.array([
        []
    ])

class SBAgent(MemberBase):
    def __init__(
        self,
        name, 
        id,
        surviver_id,
        rng,
        decision_num = 3,
        mixed_strategy = False,
    ):
        super().__init__(name, id, surviver_id, rng)

        self.fitness = 0

        self.decision_num = decision_num
        self.mixed_strategy = mixed_strategy
        self.strategy = rng.dirichlet(np.ones(self.decision_num), size=1)[0]

        self.parent = None
        self.ancestor = self

    def decision(self):
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
        rng: np.random.Generator
    ) -> "SBAgent":
        child = SBAgent(name, id, surviver_id, rng)

        child.parent = parent
        child.ancestor = parent.ancestor
        child.strategy = parent.strategy

        return child

    def mutation(self):
        self.strategy = self._rng.dirichlet(np.ones(self.decision_num), size=1)[0]
        

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
            mixed_strategy=mixed_strategy,
        ) for i in range(self.init_member_num)]
        self.all_members: List[SBAgent] = self._backup_member_list(self.init_members)
        self.current_members: List[SBAgent] = self._backup_member_list(self.init_members)

        self.payoff_matrix = pay_off_matrix

    def single_game(self, player_1: SBAgent, player_2: SBAgent):
        if player_1 == player_2:
            return
        
        player_1_strategy = player_1.decision()
        player_2_strategy = player_2.decision()

        player_1.fitness += self.payoff_matrix[player_1_strategy, player_2_strategy]
        player_2.fitness += self.payoff_matrix[player_2_strategy, player_1_strategy]

    def multiple_game(self, game_num, cpu_num=1):
        # # use multiprocessing.Pool
        # # it doesn't work because when using Pool, the agent is not the same as the one 
        # # in the self.current_members, so it's not updated
        # game_member_list = []
        # for _ in range(game_num):
        #     game_member_list.append(
        #         (self.random_pick(1)[0], self.random_pick(1)[0])
        #     )

        # # with Pool(cpu_num) as p: 
        #     for _ in range(game_num):
        #         p.starmap(
        #             self.single_game, 
        #             game_member_list,
        #         )

        # use for loop
        for _ in range(game_num):
            self.single_game(*self.random_pick(2))

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