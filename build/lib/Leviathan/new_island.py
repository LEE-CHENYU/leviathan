class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        self.chain_of_action1 = [('attack', 1, 4), ('offer', 2, 1), ('offer', 1, 2)]
        self.chain_of_action2 = [('attack', 4, 1), ('attack', 3, 1)]
        self.chain_of_action3 = [('attack', 1, 4), ('offer', 2, 1), ('offer', 1, 2)]

        self.chain_of_action = [self.chain_of_action1, self.chain_of_action2, self.chain_of_action3]

    def offer(self, member_1, member_2, parameter_influence):
        super()._offer(member_1, member_2, parameter_influence)
        
    def attack(self, member_1, member_2):
        super()._attack(member_1, member_2)

    def bear(self, member_1, member_2):
        super()._bear(member_1, member_2)

    def execute(self):
        cnt = 0
        
        while True:

            j = 0
            
            for i in range(len(self.chain_of_action)):
                if len(self.chain_of_action[i]) > cnt:
                    chain = self.chain_of_action[i][cnt]
                    if chain[0] == "attack":
                        self.attack(self.all_members[chain[1]], self.all_members[chain[2]])
                    elif chain[0] == "offer":
                        self.offer(self.all_members[chain[1]], self.all_members[chain[2]], True)
                    elif chain[0] == "bear":
                        self.bear(self.all_members[chain[1]], self.all_members[chain[2]])
                else:
                    j += 1
                    
            cnt += 1
            
            if j >= len(self.chain_of_action):
                break

def main():
    
    # from Leviathan.Island import Island
    # from Leviathan.Member import Member
    # from Leviathan.Analyzer import Analyzer
    from time import time
    # from Leviathan.Land import Land
    from utils import save
    import os

    rng = np.random.default_rng()
    path = save.datetime_dir("../data")
    exec = IslandExecution(5, (5, 5), path, 2023)
    IslandExecution._RECORD_PERIOD = 1
    Member._DECISION_BACKEND = 'inner product'
    Member._PARAMETER_INFLUENCE = 0

    action_prob = 0.5
    exec.new_round()
    exec.get_neighbors()
    
    exec.execute()
    exec.log_status(action=True, log_instead_of_print=False)

if __name__ == "__main__":
    main()
