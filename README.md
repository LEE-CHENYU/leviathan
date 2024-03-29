# Leviathan: Exploring the Emergence of Complex Systems through Individual Decisions and Relationships

探究个体决策和简单关系如何引发复杂系统的涌现。

Leviathan is a fascinating simulation project that delves into the intricacies of how individual decision-making and simple relationships can give rise to complex systems. Drawing inspiration from Thomas Hobbes' concept of the Leviathan, this project aims to investigate the emergence of complex behaviors and social structures from the interactions of autonomous agents.

## Overview

In Leviathan, players take on the role of survivors on a 2D grid-based island. Each survivor must make strategic decisions to manage their resources, form alliances, and ensure their long-term survival. The game is turn-based, and in each turn, survivors can choose to perform actions such as attacking other survivors, offering resources, reproducing, or allowing others to pass through their territory.

The decision-making process of each survivor is influenced by a set of parameters that determine their inclinations towards certain actions based on their own attributes and the attributes of potential targets. These parameters are encoded as gene values and are subject to evolutionary pressures as the simulation progresses.

## Features

- **Turn-based Gameplay**: Survivors take turns making decisions and performing actions on the island.
- **Resource Management**: Survivors must manage their vitality, cargo (food), and territory to ensure their survival.
- **Social Interactions**: Survivors can form alliances, engage in resource exchange, and reproduce to create new survivors.
- **Evolutionary Decision-Making**: Each survivor's decision-making process is guided by gene values that determine their predispositions towards certain actions. These gene values evolve over time based on the outcomes of their decisions.
- **Emergent Behavior**: The complex interactions among survivors give rise to emergent behaviors and strategies that can be analyzed and studied.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/leviathan.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the simulation:
   ```
   python main.py
   ```

## Usage

1. Configure the simulation parameters in the `config.py` file, such as the island size, initial population, and evolutionary settings.

2. Run the simulation using the command:
   ```
   python main.py
   ```

3. Observe the simulation progress and analyze the emerging behaviors and strategies of the survivors.

4. Experiment with different parameter settings and initial conditions to explore various scenarios and outcomes.

## Contributing

Contributions to Leviathan are welcome! If you find any bugs, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request. Make sure to follow the project's coding style and guidelines.

## Acknowledgments

- The Leviathan simulator was inspired by the concepts of multi-agent systems, game theory, and evolutionary algorithms.
- Special thanks to the contributors and reviewers who have helped improve the project.

# 构想

## 参数、选择函数、规则、进化、死亡

参数就是游戏中代表个体特征的数，定义了环境以及自身；
选择函数会让个体根据环境以及自身的参数进行选择；
规则规定了选择后对环境与自身的参数进行的改动；
进化（遗传、变异）确定了个体选择之外的参数改动；
死亡过程让改动变得有意义

依靠随机过程，而不是给定逻辑；不需要思考函数的参数是否合理，只需要创造环境对于参数进行选择（如何能让选择过程加快？）
- 交易、思想传播
- 交配繁衍，遗传变异
- 亲密度再定义
- 自发秩序形成
- 主动思考，参数遍历
- 环境变化
- 死亡
- 添加约束条件，给出方向，加快进化
- 无限迭代

交易：一个角色给另一个角色一定资源，另一个角色可以选择保留或者反赠一定量。交易额可以作为包含自身所有参数的一个函数的系数或者一个项。函数形式和系数等由随机过程自动生成，如果交易能够完成，则函数本身系数有机会在两方之间进行传播（模拟社会中彼此互动个体变得更相似）。

交配：每隔若干回合角色通过一个函数与其他角色进行交配，交配过程生成一个包含部分母体信息的新个体（模仿减数分裂？）。新个体创造过程会出现随机变异。可以制作表示个体亲子关系的三维图。

亲密度：亲密度可以再定义为个体间的相似程度，也是达成互动（助战、交易、交配）的成功率的指数。所以也许只需要通过函数系数的方式来体现？

秩序形成：不再需要分配策略、领导选举，完全依靠自发决定。但保留尊重值作为一个函数中的一个项，因为尊重与打架胜率挂钩。

主动思考：想办法创造一个算法遍历整个程序中的每个参数，并随机生成任何可能的函数式。参数是环境，函数是对环境的适应。

环境变化：环境是个体间的互动规则，需要一个可以创造规则的环境类。个体可以选择是否接受某一个规则。比如说是否进行交易，是否选择抢劫。规则规定的行为可以是跨回合的。

死亡：资源约束强制死亡的发生。

约束条件：将给定阈值改为引导性的条件，比如为某一方向规定很大或者更小的发生概率，但不去人为排除任何选项。

# 框架

## 生存单位：人
- 基本属性：
  - 生产力
  - 决策参数
  - 父母与孩子
- 状态：
  - 血量
  - 食物存储
  - 年龄
  - 关系（记忆）

## 基本关系
- 人与人相互之间的关系由相互行动的记忆决定
- 记忆包含：
  - 给予/收到食物的数量
  - 战斗中的被打
  - 战斗中的被帮助

## 行动
- 人的行动包含自我行动以及相互行动：
  - 自我行动：生产，消费
  - 相互行动：交易，战斗，生育，模仿
- 行动与否，行动的方式由决策函数描述
  - 每个决策（暂时）由线型函数与阈值（$ax + by + \dots>1$）决定
  - 函数的参数（$a, b, \dots$）为知识，由遗传和学习影响
  - 函数的输入（$x, y, \dots$）为相互关系和双方状态
- 行动的结果
  - 改变记忆
  - 改变个人状态
  - （生育）产生新个体

# 实现方式

## 数据结构
- Member类：
  - 描述个体
  - 记录基本属性
  - 记录状态
- Island类：
  - 描述集体，控制模拟轮次

## 模拟方式
1. 初始化：
   1. 产生$N$位Member，随机设定初始属性和状态，随机设定相互关系，随机设置决策函数参数
2. 生产
   1. 根据生产力，增加食物存储
3. 战斗
   1. 随机分组、组内排序。
   2. 按概率设定某组相互开战与否（为了减少运算消耗）
   3. 在开战的组内，遍历组员。根据【攻击决策】函数，选出所有攻击者与被攻击者的组合
   4. 双方互相攻击，互相造成对方扣除与自身生命值相关的血量；双方互相偷盗对方的财产，数额与自身生命值相关
4. 交易与交流
   1. 随机分组、组内排序。
   2. 根据【给予决策】函数，选出一个（或零个）给予对象，给予与决策函数相关的仓库数额（为了避免bug，此数额要小于等于仓库存储量）
   3. 【给予决策】函数：需要考虑双方的关系网，如把对其他人记忆的内积作为输入。
   4. 被给予者的记忆会被帮助者影响，记忆改变为两人的均值
5. 消费
   1. 计算消耗量。消耗量会随着年龄逐步提升
   2. 从血量中扣除消耗量，若血量小于零则记为死亡
   3. 从仓库中吃食物回满血
6. 生育
   1. 择出满足年龄条件的人
   2. 随机分组，组内排序。
   3. 每组内便利，根据【生育决策】函数，判断互相好感，选择父母
   4. 判断双方是否满足生育条件（血量和仓库之和）
   5. 父母扣除固定仓库数，仓库不足时扣除血量。
   6. 产生孩子。设定孩子年龄（0），父母。孩子随机继承父母的基本属性与决策参数，添加**少许**随机浮动。孩子的初始血量为固定值（小于父母消耗值），存储……
7. （模仿）
8. 重复2～
