# Leviathan

## 功能概述 (Functionality Overview)

Leviathan 是一个基于智能体的社会演化模拟框架，专注于模拟智能体间的互动、资源分配和关系网络。框架的核心功能包括：

- **智能体决策系统**：基于内置决策算法或大型语言模型的智能体行为决策
- **地形与资源系统**：模拟土地分配、资源生产与消耗
- **关系网络**：追踪智能体间的各种关系（敌对、互惠等）
- **人口动态**：模拟人口增长、衰退与世代更替
- **数据收集与分析**：记录模拟过程并提供分析工具

Leviathan is an agent-based social evolution simulation framework that focuses on simulating interactions between agents, resource allocation, and relationship networks. The core functionalities include:

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

## 核心方法 (Core Methods)

### Island 类方法 (Island Class Methods)

#### 基础操作 (Basic Operations)
- `new_round()`: 开始新一轮模拟
- `save_current_island(path)`: 保存当前模拟状态
- `load_island(path)`: 加载模拟状态

#### 智能体交互 (Agent Interactions)
- `_attack(member_1, member_2)`: 攻击行为
- `_offer(member_1, member_2)`: 资源给予
- `_bear(member_1, member_2)`: 生育行为
- `_offer_land(member_1, member_2)`: 土地给予

#### 集体行为 (Collective Behaviors)
- `produce()`: 所有智能体进行资源生产
- `consume()`: 所有智能体消耗资源
- `fight(prob_to_fight)`: 执行攻击行为
- `trade(prob_to_trade)`: 执行交易行为
- `reproduce(prob_of_reproduce)`: 执行繁殖行为

#### 分析与记录 (Analysis and Recording)
- `generate_decision_history()`: 生成决策历史
- `record_historic_ratio()`: 记录历史比率
- `log_status()`: 输出当前状态

### Member 类方法 (Member Class Methods)

- `decision(decision_name, object, island)`: 做出决策
- `produce()`: 生产资源
- `consume()`: 消耗资源
- `recover()`: 恢复生命值
- `parameter_absorb(contributor_list, weight_list)`: 参数学习

### IslandExecution 类方法 (IslandExecution Class Methods)

- `execute()`: 执行所有预设行动
- `decision(member_id)`: 获取智能体决策
- `append_to_action_board(result)`: 添加行动到行动面板

## 使用方法 (Usage Methods)

### 基本使用流程 (Basic Usage Flow)

1. 初始化 Island 或 IslandExecution 实例
2. 配置模拟参数和环境
3. 运行模拟循环
4. 分析结果

```python
from Leviathan.Island import Island
from Leviathan.islandExecution import IslandExecution

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
    
    # 记录和日志
    if i % 10 == 0:
        island.log_status()
```

### 使用 IslandExecution 控制行动 (Using IslandExecution to Control Actions)

```python
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

# 或者通过决策系统添加行动
for member in island_exec.current_members:
    action = island_exec.decision(member.id)
    island_exec.append_to_action_board(action)
```

### 分析和可视化结果 (Analysis and Visualization of Results)

```python
import pandas as pd
import matplotlib.pyplot as plt

# 获取智能体特征
features = island.get_current_member_features()

# 绘制人口变化
plt.figure(figsize=(10, 6))
plt.plot(island.record_population)
plt.title('人口变化')
plt.xlabel('回合')
plt.ylabel('人口数量')
plt.savefig('./results/population.png')

# 分析社会网络
relationship_matrix = island.relationship_dict['benefit']
plt.figure(figsize=(8, 8))
plt.imshow(relationship_matrix, cmap='coolwarm')
plt.colorbar()
plt.title('互惠关系网络')
plt.savefig('./results/relationship_network.png')
```

## 环境要求 (Environment Requirements)

- Python 3.7+
- 依赖库: numpy, pandas, matplotlib
- 可选依赖: openai (用于GPT决策)

## 注意事项 (Notes)

- 大规模模拟可能需要较长运行时间
- 使用GPT决策功能需要配置有效的API密钥
- 随机种子设置可以确保模拟的可重复性
- 建议在分析前保存模拟数据，以便后续处理 