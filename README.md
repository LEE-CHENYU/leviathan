# Leviathan

Investigating if individual decisions and simple relationships can converge to form a complex system in the manner of Hobbesian Leviathan.

试图通过模型中分布式的简单个体选择模拟类似社会秩序的复杂系统。

## 项目概述 (Project Overview)

Leviathan 是一个基于智能体的社会演化模拟框架，探索简单个体决策如何收敛形成复杂社会系统。该框架通过模拟智能体间的互动、资源分配和关系网络，研究社会结构的自发形成过程。

Leviathan is an agent-based social evolution simulation framework that explores how simple individual decisions converge to form complex social systems. The framework studies the spontaneous formation of social structures through simulating interactions between agents, resource allocation, and relationship networks.

## 核心功能 (Core Functionality)

- **智能体决策系统**：基于内置决策算法或大型语言模型的智能体行为决策
- **地形与资源系统**：模拟土地分配、资源生产与消耗
- **关系网络**：追踪智能体间的各种关系（敌对、互惠等）
- **人口动态**：模拟人口增长、衰退与世代更替
- **数据收集与分析**：记录模拟过程并提供分析工具

- **Agent Decision System**: Agent behavior decisions based on built-in algorithms or large language models
- **Terrain and Resource System**: Simulation of land allocation, resource production, and consumption
- **Relationship Network**: Tracking various relationships between agents (hostility, reciprocity, etc.)
- **Population Dynamics**: Simulation of population growth, decline, and generational changes
- **Data Collection and Analysis**: Recording simulation processes and providing analysis tools

## 主要接口 (Main Interfaces)

### Island 类 (Island Class)

核心模拟环境类，管理智能体、土地和各种交互：

```python
Island(
    init_member_number: int,     # 初始成员数量
    land_shape: Tuple[int, int], # 土地形状 (行, 列)
    save_path: str,              # 保存路径
    random_seed: Optional[int] = None, # 随机种子
)
```

### Member 类 (Member Class)

代表单个智能体的类，包含属性、决策逻辑和行为：

```python
Member(
    name: str,                   # 智能体名称
    id: int,                     # 唯一ID
    surviver_id: int,            # 存活者ID
    rng: np.random.Generator     # 随机数生成器
)
```

### Land 类 (Land Class)

管理土地所有权和地形特性：

```python
Land(
    shape: Tuple[int, int],      # 地形形状 (行, 列)
)
```

### IslandExecution 类 (IslandExecution Class)

扩展 Island 类以支持控制和执行智能体行动：

```python
IslandExecution(
    init_member_number: int,     # 初始成员数量
    land_shape: Tuple[int, int], # 土地形状 (行, 列)
    save_path: str,              # 保存路径
    random_seed: Optional[int] = None, # 随机种子
    action_board: List[List[Tuple[str, int, int]]] = None # 行动面板
)
```

## 使用方法 (Usage Methods)

### 基本使用流程 (Basic Usage Flow)

```python
from Leviathan.Island import Island

# 创建基础模拟
island = Island(
    init_member_number=10,
    land_shape=(20, 20),
    save_path="./simulation_results",
    random_seed=42
)

# 运行多轮模拟
for i in range(100):
    island.new_round()
    island.produce()       # 资源生产
    island.fight(0.3)      # 战斗（0.3概率）
    island.trade(0.5)      # 交易（0.5概率）
    island.reproduce(0.2)  # 繁殖（0.2概率）
    island.consume()       # 资源消耗
```

### 使用 IslandExecution 控制行动 (Using IslandExecution)

```python
from Leviathan.islandExecution import IslandExecution

# 预设行动序列
action_board = [
    [('attack', 1, 2), ('offer', 3, 4)],  # 第一轮行动
    [('reproduce', 1, 5), ('attack', 2, 3)]  # 第二轮行动
]

# 创建可执行模拟
island_exec = IslandExecution(
    init_member_number=10,
    land_shape=(20, 20),
    save_path="./simulation_results",
    random_seed=42,
    action_board=action_board
)

# 执行预设行动
island_exec.execute()
```

### 使用 MetaIsland 增强功能 (Using MetaIsland Enhanced Features)

MetaIsland 扩展了基础 Leviathan 功能，增加了 LLM 驱动的智能体决策和机制修改能力：

```python
import asyncio
from MetaIsland.metaIsland import IslandExecution

async def main():
    # 创建 IslandExecution 实例
    island = IslandExecution(
        init_member_number=10,
        land_shape=(20, 20),
        save_path="./simulation_results"
    )
    
    # 执行多轮模拟
    for _ in range(5):
        island.new_round()
        
        # 执行智能体决策和行动
        for member in island.current_members:
            await island.agent_code_decision(member.id)
            await island.agent_mechanism_proposal(member.id)
            await island.analyze(member.id)
        
        # 执行所有决策和机制修改
        island.execute_code_actions()
        island.execute_mechanism_modifications()

# 运行模拟
if __name__ == "__main__":
    asyncio.run(main())
```

## 设计理念 (Design Philosophy)

本项目探索通过以下机制形成社会秩序：

- **交易与思想传播**：智能体间交换资源和学习彼此决策模式
- **遗传与变异**：通过繁殖传递和变异决策参数
- **关系网络形成**：智能体互动形成复杂关系网络
- **自发秩序**：无需中央控制，通过个体决策形成社会秩序
- **资源约束与竞争**：有限资源引发竞争和合作行为

This project explores the formation of social order through mechanisms including:

- **Trade and Idea Propagation**: Agents exchange resources and learn from each other's decision patterns
- **Inheritance and Mutation**: Decision parameters transmitted and mutated through reproduction
- **Relationship Network Formation**: Complex relationship networks formed through agent interactions
- **Spontaneous Order**: Social order emerges from individual decisions without central control
- **Resource Constraints and Competition**: Limited resources drive competitive and cooperative behaviors

## 环境要求 (Environment Requirements)

- Python 3.7+
- 依赖库: numpy, pandas, matplotlib
- 高级功能依赖: openai, aisuite

## E2E Smoke Test (LLM required)

The end-to-end smoke test (`python scripts/run_e2e_smoke.py`) requires a live LLM.
Offline stubs are not supported for evaluation.

OpenRouter example:

```bash
export OPENROUTER_API_KEY="..."
export E2E_PROVIDER=openrouter
export E2E_MODEL="minimax/minimax-m2.1"  # optional override
python scripts/run_e2e_smoke.py
```

OpenAI-compatible gateway or OpenAI example:

```bash
export OPENAI_API_KEY="..."
export E2E_PROVIDER=openai
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your gateway, e.g. http://localhost:8000/v1
export E2E_MODEL="gpt-5.2"  # optional override
python scripts/run_e2e_smoke.py
```

If you prefer repo config over env vars, you can set `base_url` under `default` (for the
default provider) or `benchmark` (for OpenAI fallback) in `config/models.yaml`. The e2e
smoke runner and `scripts/llm_access_check.py` will use it when `OPENAI_BASE_URL` /
`OPENROUTER_BASE_URL` are not set.

If preflight fails, check the diagnostics in
`execution_histories/e2e_smoke/latest_preflight.json` along with the stderr hints.

## 详细文档 (Detailed Documentation)

更详细的文档可查看以下目录：
- [Leviathan 基础框架文档](/Leviathan/README.md)
- [MetaIsland 增强功能文档](/MetaIsland/README.md)

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
