from Leviathan.Island import Island

from sys import stdin 
from select import select
import os

class Engine():
    def __init__(
        self,
        init_member_num,
        land_shape,
        random_seed,
        record_path,
    ) -> None:
        self.island = Island(
            init_member_num,
            land_shape,
            random_seed
        )
        
        Island._RECORD_PERIOD = 100

        if not os.path.exists(record_path):
            os.mkdir(record_path)
        self.record_path = record_path
        
    def run(self):
        while self.island.current_member_num > 1:
            self.island.produce()
            self.island.fight()
            self.island.consume()
            self.island.trade()
            self.island.reproduce(group_size=10)
            self.island.new_round(record_path=self.record_path)

            # if self.island.current_round % self.island._RECORD_PERIOD == 0:
            #     self._interupt(1000)

    def _interupt(self, wait_time):
        get_input, _, _ = select([stdin], [], [], 0.1)
        if get_input and stdin.readline().strip() == "p":
            print("按回车继续")
            get_input, _, _ = select([stdin], [], [], wait_time)
        else:
            return

    def save(self):
        pass
