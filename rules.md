# 利维坦框架

## 生存单位：人
- 基本属性：
  - 生产力
- 状态：
  - 血量
  - 食物存储

## 基本关系
- 人与人相互之间的关系由相互行动的记忆决定
- 记忆包含：
  - 给予/收到食物的数量
  - 战斗中的帮助与被帮助

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
- Island类：
  - 描述集体，控制模拟轮次

## 模拟方式
1. 初始化：
2. 生产
3. 
4. 重复2～

