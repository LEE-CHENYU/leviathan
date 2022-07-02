import numpy as np

class Member():

    # 初始值
    MIN_PRODUCTIVITY, MAX_PRODUCTIVITY = 10, 20     # 生产力属性
    INIT_MIN_VIT, INIT_MAX_VIT = 10, 90             # 初始血量
    INIT_MIN_CARGO, INIT_MAX_CARGO = 0, 100         # 初始食物存储
    INIT_MIN_AGE, INIT_MAX_AGE = 10, 1000           # 初始年龄
    # CONSUMPTION 
    
    # 决策参数们


    @classmethod
    def born(cls, parent_1, parent_2):
        pass

    def __init__(
        self, 
        name: str, 
        id: int, 
        surviver_id: int,
        rng: np.random.Generator
        ) -> None:

        # 随机数生成器
        self._rng = rng

        # 姓名
        self.name = name
        self.id = id
        self.surviver_id = surviver_id      # Island.current_members中的编号

        # 亲人链表
        self.parent_1 = None
        self.parent_2 = None
        self.child = []

        #####
        # 决策参数
        #####

        # 生产相关的属性和状态
        self.productivity = self._rng.uniform(Member.MIN_PRODUCTIVITY, Member.MAX_PRODUCTIVITY)
        self.vitality = self._rng.uniform(Member.INIT_MIN_VIT, Member.INIT_MAX_VIT)
        self.cargo = self._rng.uniform(Member.INIT_MIN_CARGO, Member.INIT_MAX_CARGO)
        self.age = int(self._rng.uniform(Member.INIT_MIN_AGE, Member.INIT_MAX_AGE))

    def __str__(self):
        """重载print函数表示"""
        return f"{self.name}({self.id})"

    def __repr__(self):
        """重载其他print形式的表示"""
        return self.__str__()
    
    @property
    def strength(self) -> float:
        """
        战斗力：每次攻击造成的伤害
        """
        pass

    @property
    def consumption(self) -> float:
        """
        每轮消耗量
        """
        pass


    #########################################################################
    ################################## 动作 ################################## 
    def _generate_decision_inputs(self, object, island):
        pass

    def _decision(self, parameters, inputs) -> float:
        return np.sum(parameters * inputs)

    def produce(self):
        self.cargo += self.productivity

    

