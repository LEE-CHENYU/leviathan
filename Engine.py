from Island import Island

from time import sleep
from sys import stdin 
from select import select

class Engine():
    def __init__(
        self,
        init_member_num,
        random_seed,
    ) -> None:
        self.island = Island(
            init_member_num,
            random_seed
        )
        pass

    def save_seed(self):
        """设置并保存随机种子"""
        pass

    def run(self):
        while self.island.current_member_num > 1:
            self.island.produce()
            self.island.fight()
            self.island.consume()
            self.island.trade()
            self.island.reproduce(group_size=10)
            self.island.new_round()

            if self.island.current_round % self.island._RECORD_PERIOD == 0:
                self.interupt(1000)

    def interupt(self, wait_time):
        get_input, _, _ = select([stdin], [], [], 0.01)
        if get_input and stdin.readline().strip() == "p":
            print("按回车继续")
            get_input, _, _ = select([stdin], [], [], wait_time)
        else:
            return


    def save(self):
        pass

engine = Engine(50, 2022)
engine.run()