# MetaIsland

## 功能概述 (Functionality Overview)

MetaIsland 是一个多智能体社会模拟框架，提供了创建虚拟社会环境并模拟智能体交互的能力。系统包含以下核心功能：

- **基于LLM的智能体决策系统**：利用大型语言模型进行复杂决策
- **动态社会机制**：允许智能体提出和修改社会规则与机制
- **关系网络模拟**：模拟智能体之间的关系、交互和资源分配
- **代码生成与执行**：智能体可以生成并执行代码，影响模拟环境
- **分析与记录系统**：记录和分析智能体行为和系统演化

MetaIsland is a multi-agent social simulation framework that enables the creation of virtual social environments and agent interactions. The system includes the following core functionalities:

- **LLM-based Agent Decision System**: Utilizes large language models for complex decision-making
- **Dynamic Social Mechanisms**: Allows agents to propose and modify social rules and mechanisms
- **Relationship Network Simulation**: Simulates relationships, interactions, and resource allocation between agents
- **Code Generation and Execution**: Agents can generate and execute code that affects the simulation environment
- **Analysis and Recording System**: Records and analyzes agent behavior and system evolution

## 主要接口 (Main Interfaces)

### IslandExecution 类 (IslandExecution Class)

主要的模拟执行类，继承自基础 Island 类：

```python
IslandExecution(
    init_member_number: int,    # 初始成员数量
    land_shape: Tuple[int, int], # 土地形状 (行, 列)
    save_path: str,             # 保存路径
    random_seed: Optional[int] = None, # 随机种子
    action_board: List[List[Tuple[str, int, int]]] = None, # 行动面板
    agent_modifications: dict = None # 智能体修改设置
)
```

### 核心方法 (Core Methods)

#### 环境控制 (Environment Control)
- `new_round()`: 开始新一轮模拟
- `save_execution_history()`: 保存执行历史
- `print_agent_performance()`: 打印智能体表现数据

#### 智能体行动 (Agent Actions)
- `async agent_code_decision(member_id)`: 生成智能体决策代码
- `async agent_mechanism_proposal(member_id)`: 生成机制提案代码
- `execute_code_actions()`: 执行所有智能体代码行动
- `execute_mechanism_modifications()`: 执行机制修改

#### 分析功能 (Analysis Functions)
- `async analyze(member_id)`: 分析智能体行为和系统状态
- `get_current_member_features()`: 获取当前成员特征

#### 通信系统 (Communication System)
- `send_message(sender_id, recipient_id, message)`: 发送信息
- `print_agent_messages()`: 打印智能体消息

## 使用方法 (Usage Methods)

### 基本使用流程 (Basic Usage Flow)

1. 初始化 IslandExecution 实例
2. 执行模拟循环
3. 分析结果

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
        
        # 保存执行历史
        island.save_execution_history()
    
    # 打印结果
    island.print_agent_performance()
    island.print_agent_messages()

# 运行模拟
if __name__ == "__main__":
    asyncio.run(main())
```

### 高级功能使用 (Advanced Features)

#### 自定义机制修改 (Custom Mechanism Modifications)

可以在运行期间通过智能体提出的机制修改来改变系统行为：

```python
# 机制修改代码示例
def propose_modification(execution_engine):
    # 添加新的交互机制
    modification = {
        "type": "add_method",
        "name": "trade_resources",
        "code": """
        def trade_resources(self, member_1, member_2, resource_amount):
            # 实现资源交易逻辑
            if member_1.storage >= resource_amount:
                member_1.storage -= resource_amount
                member_2.storage += resource_amount
                self.relationship_modify('benefit', member_1, member_2, resource_amount)
                return True
            return False
        """
    }
    return modification
```

#### 分析和可视化 (Analysis and Visualization)

使用内置分析功能来理解模拟结果：

```python
# 分析代码示例
def analyze_simulation(execution_engine):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 获取智能体表现数据
    performance_data = pd.DataFrame(execution_engine.performance_history)
    
    # 可视化关系网络
    relationship_matrix = execution_engine.relationship_dict['benefit']
    plt.figure(figsize=(10, 8))
    plt.imshow(relationship_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('智能体间互惠关系')
    plt.savefig('./results/relationship_network.png')
    
    return {
        "performance_trends": performance_data.describe().to_dict(),
        "key_insights": ["资源分配集中于特定智能体", "关系网络显示社区形成"]
    }
```

## 环境要求 (Environment Requirements)

- Python 3.8+
- 依赖库: numpy, pandas, openai, asyncio, aisuite
- 环境变量配置 (.env 文件)

## 注意事项 (Notes)

- 智能体代码执行在受控环境中，确保安全性
- 大型模拟可能消耗大量计算资源
- 建议先进行小规模测试，再扩展到更大规模的模拟 