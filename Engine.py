from Island import Island

from time import sleep
from sys import stdin 
from select import select
import os

class Engine():
    def __init__(
        self,
        init_member_num,
        random_seed,
        record_path,
    ) -> None:
        self.island = Island(
            init_member_num,
            random_seed
        )
        
        Island._RECORD_PERIOD = 100

        self.record_path = record_path
        if not os.path.exists(self.record_path):
            os.mkdir(self.record_path)
        else:
            raise ValueError(f"已经存在路径 {self.record_path}，请指定不存在的文件夹。")

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

engine = Engine(50, 2022, record_path="test6/")
engine.run()