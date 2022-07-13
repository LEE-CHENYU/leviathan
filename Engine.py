from Island import Island

from time import sleep

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
            # sleep(10)

    def save(self):
        pass

engine = Engine(100, 2022)
engine.run()