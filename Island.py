import numpy as np
import pandas as pd
from Member import Member
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from time import time

def _requirement_for_reproduction(
    member_1: Member, 
    member_2: Member
) -> bool:
    return (member_1.vitality + member_1.cargo
        + member_2.vitality + member_2.cargo
    ) >= Island._REPRODUCE_REQUIREMENT

def _requirement_for_offer(
    member_1: Member, 
    member_2: Member
) -> bool:
    return member_1.cargo * Member._MIN_OFFER_PERCENTAGE >= Member._MIN_OFFER

class Island():
    _MIN_VICTIM_MEMORY, _MAX_VICTIM_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0
    _MIN_BENEFIT_MEMORY, _MAX_BENEFIT_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0

    _REPRODUCE_REQUIREMENT = 150                            # 生育条件：双亲血量和仓库之和大于这个值
    assert _REPRODUCE_REQUIREMENT > Member._CHILD_VITALITY

    # 记录/输出周期
    _RECORD_PERIOD = 100

    def __init__(
        self, 
        init_member_number: int,
        random_seed: int = None
    ) -> None:

        # 设置并记录随机数种子
        self._create_from_file = False
        self._file_name = ""
        if random_seed is not None:
            self._random_seed = int(random_seed)
        else:
            self._random_seed = int(time())
        self._rng = np.random.default_rng(self._random_seed)

        # 初始人数，当前人数
        self._NAME_LIST = self._rng.permutation(np.loadtxt("./name_list.txt", dtype=str))

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # 初始人物列表，全体人物列表，当前人物列表
        self.init_members = [Member(self._NAME_LIST[i], id=i, surviver_id=i, rng=self._rng) for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # 初始人物关系
        # 关系矩阵M，第j行 (M[j, :]) 代表第j个主体的被动记忆（受伤/受赠……）
        # 若要修改（增减）人物关系，需要修改：self.relationship_dict, Member.DECISION_INPUT_NAMES, Member._generate_decision_inputs()
        _victim_memory = self._rng.uniform(Island._MIN_VICTIM_MEMORY, Island._MAX_VICTIM_MEMORY, size=(self.init_member_num, self.init_member_num))
        _victim_memory[_victim_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(_victim_memory, np.nan)

        _benefit_memory = self._rng.uniform(Island._MIN_BENEFIT_MEMORY, Island._MAX_BENEFIT_MEMORY, size=(self.init_member_num, self.init_member_num))
        _benefit_memory[_benefit_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(_benefit_memory, np.nan)

        self.relationship_dict = {
            "victim": _victim_memory,
            "benefit": _benefit_memory
        }
        assert len(self.relationship_dict) == len(Member._RELATION_SCALES), "关系矩阵数量和关系矩阵缩放量数量不一致"

        # 记录动作 （每Island._RECORD_PERIOD输出、清空一次）
        self.record_attack = {}
        self.record_benefit = {}
        self.record_born = []
        self.record_death = []

        # 记录状态 （每Island._RECORD_PERIOD向末尾增append一个0）
        self.record_total_production = [0]
        self.record_total_consumption = [0]
        self.record_total_attack = [0]
        self.record_total_benefit = [0]

        # 回合数
        self.current_round = 0


    #########################################################################
    ################################ 基本操作 ################################# 
    def _backup_member_list(
        self, 
        member_list: List[Member]
    ) -> List[Member]:
        """复制member_list"""
        return [member for member in member_list]

    def _member_list_append(
        self, 
        append: List[Member] = [], 
        appended_rela_rows: np.ndarray = [], 
        appended_rela_columnes: np.ndarray = []
    ) -> None:
        """
        向current_members，all_members增加人物，
        增加current_member_num，
        修改relationships矩阵，
        修改人物surviver_id
        """
        appended_num = len(append)
        prev_member_num = self.current_member_num

        if not isinstance(appended_rela_columnes, np.ndarray):
            raise ValueError("关系矩阵增添的列应该是ndarray类型")
        if not isinstance(appended_rela_rows, np.ndarray):
            raise ValueError("关系矩阵增添的行应该是ndarray类型")
        assert appended_rela_columnes.shape == (prev_member_num, appended_num), "输入关系列形状不匹配"
        assert appended_rela_rows.shape == (appended_num, prev_member_num), "输入关系行形状不匹配"

        # 向列表中增加人物
        for member in append:
            member.surviver_id = self.current_member_num
            self.current_members.append(member)
            self.all_members.append(member)

            self.current_member_num += 1

        # 记录出生
        self.record_born = self.record_born + append

        # 修改关系矩阵
        for key in self.relationship_dict.keys():
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp_old = self.relationship_dict[key].copy()
            tmp_new = np.zeros((self.current_member_num, self.current_member_num))
            
            tmp_new[:prev_member_num, :prev_member_num] = tmp_old
            tmp_new[:prev_member_num, prev_member_num:] = appended_rela_columnes
            tmp_new[prev_member_num:, :prev_member_num] = appended_rela_rows
            np.fill_diagonal(tmp_new, np.nan)

            self.relationship_dict[key].resize(
                (self.current_member_num, self.current_member_num), 
                refcheck=False
            )
            self.relationship_dict[key][:] = tmp_new          # 仅修改数组内容，不修改列表里的数组地址

        return

    def _member_list_drop(
        self, 
        drop: List[Member] = []
    ) -> None:
        """
        从current_members删除人物，
        减少current_member_num，
        修改relationships矩阵，
        重新修改全体人物surviver_id
        """
        drop_id = np.array([member.id for member in drop])            # 校对id，确保正确删除
        drop_sur_id = np.array([member.surviver_id for member in drop])

        assert not (drop_sur_id == None).any(), "被删除对象没有surviver_id"

        # 排序，确保正确移除
        argsort_sur_id = np.argsort(drop_sur_id)[::-1]
        drop_id = drop_id[argsort_sur_id]
        drop_sur_id = drop_sur_id[argsort_sur_id]
        
        # 从列表中移除人物
        for idx in range(len(drop_id)):
            id_to_drop = drop_id[idx]
            sur_id_to_drop = drop_sur_id[idx]
            assert self.current_members[sur_id_to_drop].id == id_to_drop, "删除对象不匹配"

            self.current_members[sur_id_to_drop].surviver_id = None

            del self.current_members[sur_id_to_drop]
            self.current_member_num -= 1

        # 记录死亡
        self.record_death = self.record_death + drop

        # 修改关系矩阵
        for key in self.relationship_dict.keys():
            # 无法直接进行赋值，需修改原数组尺寸后填入数值
            tmp = np.delete(self.relationship_dict[key], drop_sur_id, axis=0)
            tmp = np.delete(tmp, drop_sur_id, axis=1)
            
            self.relationship_dict[key].resize(
                (self.current_member_num, self.current_member_num), 
                refcheck=False
            )
            self.relationship_dict[key][:] = tmp    # 仅修改数组内容，不修改列表里的数组地址

        # 重新编号存活成员
        for sur_id in range(self.current_member_num):
            self.current_members[sur_id].surviver_id = sur_id
        
        return

    def member_list_modify(
        self, 
        append: List[Member] = [], 
        drop: List[Member] = [], 
        appended_rela_rows: np.ndarray = np.empty(0), 
        appended_rela_columnes: np.ndarray = np.empty(0)
    ) -> None:
        """
        修改member_list，先增加人物，后修改
        记录出生、死亡
        """
        if append != []:
            self._member_list_append(
                append=append, 
                appended_rela_rows=appended_rela_rows, appended_rela_columnes=appended_rela_columnes
            )
        if drop != []:
            self._member_list_drop(
                drop=drop
            )

        return

    def _overlap_of_relations(
        self, 
        principal: Member, 
        object: Member
        ) -> List[float]:
        """计算关系网内积"""

        def normalize(arr):
            """剔除nan，归一化向量"""
            arr[principal.surviver_id] = 0
            arr[object.surviver_id] = 0
            norm = np.linalg.norm(arr)
            if norm == 0:
                return 0
            else:
                return arr / norm

        overlaps = []
        for relationship in list(self.relationship_dict.values()):
            pri_row = normalize(relationship[principal.surviver_id, :].copy())
            pri_col = normalize(relationship[:, principal.surviver_id].copy())
            obj_row = normalize(relationship[object.surviver_id, :].copy())
            obj_col = normalize(relationship[:, object.surviver_id].copy())

            overlaps.append((
                np.sum(pri_row * obj_row)
                + np.sum(pri_row * obj_col)
                + np.sum(pri_col * obj_row)
                + np.sum(pri_col * obj_col)) / 4
            )
        
        return overlaps

    def _relations_w_normalize(
        self,
        principal: Member,
        object: Member
    ) -> List[float]:
        """计算归一化（tanh）后的关系矩阵元"""
        elements = []
        for relationship in list(self.relationship_dict.values()):
            elements.append(relationship[principal.surviver_id, object.surviver_id])
            elements.append(relationship[object.surviver_id, principal.surviver_id])

        elements = np.array(elements)
        return np.tanh(elements * np.repeat(Member._RELATION_SCALES, 2))

    def relationship_modify(
        self, 
        relationship_name,
        member_1: Member, 
        member_2: Member, 
        add_value: float
    ) -> None:
        """
        增加矩阵元[member_1.surviver_id, member_2.surviver_id]
        """
        assert member_1 is not member_2, "不能修改关系矩阵中的对角元素"
        relationship = self.relationship_dict[relationship_name]
        relationship[member_1.surviver_id, member_2.surviver_id] += add_value

    def _split_into_groups(
        self, 
        group_size: int = 10, 
        prob_group_in_action: float = 1.0,
        to_split = None,
        ) -> List[List[Member]]:
        """打乱，随机分组"""

        if to_split is None:
            to_split = self.current_members
            total_num = self.current_member_num
        else:
            total_num = len(to_split)

        shuffled_members = self._backup_member_list(to_split)
        self._rng.shuffle(shuffled_members)

        group_num = np.round(total_num / group_size).astype(int)
        if group_num < 1:
            group_num = 1
        idx_list_list = np.array_split(np.arange(total_num, dtype=int), group_num)

        group_list = []
        for idx_list in idx_list_list:
            group = []
            for member_idx in idx_list:
                group.append(shuffled_members[member_idx])
            
            # 每组按概率发生行为
            if self._rng.random() < prob_group_in_action:
                group_list.append(group)

        return group_list

    def _get_pairs_from_group(
        self, 
        decision_name: str,
        group: List[Member],
        other_requirements: Callable = None,
        bilateral: bool = False
    ) -> List[List[Member]]:
        """
        根据决策函数，从小组中选出人物对
        decision_name: Member.parameter_dict的keys之一
        other_requirements: 函数，输入为两个Member，输出True（通过要求）/False
        bilateral: 设为True后，决策函数的判定为双向符合
        """
        group_size = len(group)
        selected = np.zeros(group_size)
        pairs = []
        for mem_idx_1 in range(group_size):
                mem1 = group[mem_idx_1]
                for mem_idx_2 in range(mem_idx_1 + 1, group_size):
                    if not selected[mem_idx_2] and not selected[mem_idx_1]:
                        mem2 = group[mem_idx_2]

                        if other_requirements is not None:
                            if not other_requirements(mem1, mem2):
                                continue
                        if not mem1.decision(decision_name, mem2, self) > 1:
                            continue

                        if not bilateral:
                            pairs.append([mem1, mem2])
                            selected[mem_idx_1] = 1
                            selected[mem_idx_2] = 1
                            continue
                        if mem2.decision(decision_name, mem1, self) > 1:
                            pairs.append([mem1, mem2])
                            selected[mem_idx_1] = 1
                            selected[mem_idx_2] = 1

        return pairs

    def _record_actions(
        self, 
        record: Dict[Tuple, float], 
        member_1: Member, 
        member_2: Member, 
        value_1: float, 
        value_2: float = None
    ):
        try:
            record[(member_1.id, member_2.id)] += value_1
        except KeyError:
            record[(member_1.id, member_2.id)] = value_1
        if value_2 is not None:
            try:
                record[(member_2.id, member_1.id)] += value_2
            except KeyError:
                record[(member_2.id, member_1.id)] = value_2

    def save_current_island(self, path):
        current_member_df = self.current_members[0].save_to_row()
        for sur_id in range(1, self.current_member_num):
            current_member_df = pd.concat([
                current_member_df,
                self.current_members[sur_id].save_to_row()],
                axis=0
            )
        
        info_df = pd.DataFrame({
            "_create_from_file": [self._create_from_file],
            "_file_name": [self._file_name],
            "_seed": [self._random_seed],
            "init_member_num": [self.init_member_num],
            "current_member_num": [self.current_member_num],
            "current_round": [self.current_round]
        })

        relationship_df = pd.DataFrame()
        for key, rela in self.relationship_dict.items():
            rela_df = pd.DataFrame(rela, index=None, columns=None)
            relationship_df = pd.concat([relationship_df, rela_df], axis=0)

        # 本轮之前保存的动作
        attack_info = [[key[0], key[1], value] for key, value in self.record_attack.items()]
        attack_df = pd.DataFrame(
            attack_info,
            columns=["attack_1", "attack_2", "value"]
        )
        benefit_info = [[key[0], key[1], value] for key, value in self.record_benefit.items()]
        benefit_df = pd.DataFrame(
            benefit_info,
            columns=["benefit_1", "benefit_2", "value"]
        )
        born_df = pd.DataFrame(
            [member.id for member in self.record_born], 
            columns=["born"]
        )
        death_df = pd.DataFrame(
            [member.id for member in self.record_death], 
            columns=["death"]
        )
        action_df = pd.concat([attack_df, benefit_df, born_df, death_df], axis=1)
        
        current_member_df.to_csv(path + "members.csv")
        info_df.to_csv(path + "island_info.csv")
        relationship_df.to_csv(path + "relationships.csv")
        action_df.to_csv(path + "action.csv")


    #########################################################################
    ################################## 模拟 ################################## 
    def produce(self):
        """
        生产  

            1. 根据生产力，增加食物存储
        """
        for member in self.current_members:
            self.record_total_production[-1] += member.produce(self.current_member_num)

    def _attack(
        self, 
        member_1: Member, 
        member_2: Member
    ) -> None:
        # 计算攻击、偷盗值
        strength_1 = member_1.strength
        # strength_2 = member_2.strength

        steal_1 = member_1.steal
        # steal_2 = member_2.steal
        if steal_1 > member_2.cargo:
            steal_1 = member_2.cargo
        # if steal_2 > member_1.cargo:
        #     steal_2 = member_1.cargo

        # 结算攻击、偷盗
        # member_1.vitality -= strength_2
        member_2.vitality -= strength_1
        # member_1.cargo -= steal_2
        member_2.cargo -= steal_1

        # 修改关系矩阵
        # self.relationship_modify("victim", member_1, member_2, strength_2 + steal_2)
        self.relationship_modify("victim", member_2, member_1, strength_1 + steal_1)

        # 记录动作
        self._record_actions(
            self.record_attack,
            member_1,
            member_2,
            strength_1 + steal_1,
            # strength_2 + steal_2,
        )
        self.record_total_attack[-1] += strength_1 + steal_1
        # self.record_total_attack[-1] += strength_2 + steal_2

        # 结算死亡
        # if member_1.autopsy():
        #     self.member_list_modify(drop=[member_1])
        if member_2.autopsy():
            self.member_list_modify(drop=[member_2])

    def fight(
        self, 
        group_size: int = 10,
        prob_group_in_fight: float = 1.0
        ):
        """
        战斗

            1. 随机分组、组内排序。  
            2. 按概率设定某组相互开战与否（为了减少运算消耗）  
            3. 在开战的组内，遍历组员。根据【攻击决策】函数，选出所有攻击者与被攻击者的组合  
            4. 双方互相攻击，互相造成对方扣除与自身生命值相关的血量；双方互相偷盗对方的财产，数额与自身生命值相关  
            5. 若有死亡案例，更新集体列表，更新编号，更新关系矩阵  
        """
        # 打乱顺序，随机分组
        group_list = self._split_into_groups(group_size, prob_group_in_fight)

        # 从每组内选出交战双方
        for group in group_list:
            pairs = self._get_pairs_from_group(
                "attack", 
                group, 
                other_requirements=None,
                bilateral=False
            )
            for mem0, mem1 in pairs:
                self._attack(mem0, mem1)

    def _offer(
        self, 
        member_1: Member, 
        member_2: Member,
        parameter_influence: bool = True
    ) -> None:
        """member_1 给予 member_2"""
        amount = member_1.offer

        # 结算给予
        member_2.cargo += amount
        member_1.cargo -= amount

        # 修改关系矩阵
        self.relationship_modify("benefit", member_2, member_1, amount)

        # 记录
        if amount > 1e-15:
            self._record_actions(
                self.record_benefit,
                member_1,
                member_2,
                amount,
                None
            )
            self.record_total_benefit[-1] += amount

        # 被给予者的参数受到影响
        if parameter_influence:
            member_2.parameter_absorb([member_1, member_2])


    def trade(
        self,
        group_size: int = 10,
        prob_group_in_trade: float = 1.0
        ):
        """
        交易与交流

            1. 随机分组、组内排序。
            2. 根据【给予决策】函数，选出一个（或零个）给予对象，给予与决策函数相关的仓库数额（为了避免bug，此数额要小于等于仓库存储量）
            3. 【给予决策】函数：需要考虑双方的关系网，如把对其他人记忆的内积作为输入。
            4. 被给予者的记忆会被帮助者影响，记忆改变为两人的均值
        """
        # 打乱顺序，随机分组
        group_list = self._split_into_groups(group_size, prob_group_in_trade)

        # 从每组内选出交战双方
        for group in group_list:
            pairs = self._get_pairs_from_group(
                "offer", 
                group, 
                other_requirements=_requirement_for_offer,
                bilateral=False
            )
            for mem0, mem1 in pairs:
                self._offer(mem0, mem1, parameter_influence=True)

    def consume(
        self, 
    ):
        """
        消费

            1. 计算消耗量。消耗量会随着年龄逐步提升 
            2. 从血量中扣除消耗量，若血量小于零则记为死亡
            3. 从仓库中吃食物回满血
            4. 若有死亡案例，更新集体列表，更新编号，更新关系矩阵
        """
        dead_member = []
        for member in self.current_members:
            consumption = member.consume()

            # 记录
            self.record_total_consumption[-1] += consumption
            if member.autopsy():
                dead_member.append(member)
        self.member_list_modify(drop=dead_member)

        for member in self.current_members:
            member.recover()

    def _bear(
        self,
        member_1: Member, 
        member_2: Member,
    ):
        child_id = len(self.all_members)
        child_sur_id = self.current_member_num

        child = Member.born(
            member_1,
            member_2,
            self._NAME_LIST[child_id % len(self._NAME_LIST)],
            child_id,
            child_sur_id,
            self._rng
        )
        self.member_list_modify(
            append=[child],
            appended_rela_columnes=np.zeros((self.current_member_num, 1)),
            appended_rela_rows=np.zeros((1, self.current_member_num)),
        )
        
        # 计算双亲给予孩子初始血量
        vitality_base = Member._CHILD_VITALITY
        # 双方资源总量
        member_1_total = member_1.cargo + member_1.vitality
        member_2_total = member_2.cargo + member_2.vitality
        total = member_1_total + member_2_total
        # 按比例扣除损失
        member_1_give = vitality_base * member_1_total / total
        member_2_give = vitality_base * member_2_total / total
        if member_1.cargo >= member_1_give:
            member_1.cargo -= member_1_give
        else:
            member_1.vitality -= (member_1_give - member_1.cargo)
            member_1.cargo = 0
        if member_2.cargo >= member_2_give:
            member_2.cargo -= member_2_give
        else:
            member_2.vitality -= (member_2_give - member_2.cargo)
            member_2.cargo = 0
        child.vitality = vitality_base
        # 孩子记住父母的付出
        self.relationship_modify("benefit", child, member_1, member_1_give)
        self.relationship_modify("benefit", child, member_2, member_2_give)

        # 父母无条件地offer孩子一次
        self._offer(member_1, child, parameter_influence=False)
        self._offer(member_2, child, parameter_influence=False)

        # 孩子计算、恢复血量
        child.recover()

    def reproduce(
        self, 
        group_size: int = 10,
        prob_group_in_reproduce: float = 1.0
    ):
        """
        生育

            1. 择出满足年龄条件的人
            2. 随机分组，组内排序。
            3. 每组内便利，根据【生育决策】函数，判断互相好感，选择父母
            4. 判断双方是否满足生育条件（血量和仓库之和）
            5. 父母按比例扣除仓库数，总和为固定值，仓库不足时扣除血量。
            6. 产生孩子。设定孩子年龄（0），父母。孩子随机继承父母的基本属性与决策参数，添加**少许**随机浮动。孩子的初始血量为固定值（小于父母消耗值），存储……
            7. 父母无条件地offer孩子一次
            8. 若有出生案例，更新集体列表，更新编号，更新关系矩阵
        """
        # 选择年龄达标成员
        qualified_member = [member for member in self.current_members if member.is_qualified_to_reproduce]

        # 随即分组、组内排序
        group_list = self._split_into_groups(
            group_size, 
            prob_group_in_reproduce, 
            to_split=qualified_member
        )

        # 选出双亲
        for group in group_list:
            pairs = self._get_pairs_from_group(
                "reproduce", 
                group, 
                other_requirements=_requirement_for_reproduction,
                bilateral=True
            )

            for mem0, mem1 in pairs:
                self._bear(mem0, mem1)

    def new_round(self):
        # 输出内容
        if self.current_round % Island._RECORD_PERIOD == 0:
            # 动作
            print("#" * 21, f"{self.current_round:d}", "#" * 21)
            print("=" * 21, "攻击", "=" * 21)
            if self.record_attack != {}:
                for key, value in self.record_attack.items():
                    member_1 = self.all_members[key[0]]
                    member_2 = self.all_members[key[1]]
                    print(f"\t{member_1} --{value:.1f}-> {member_2}")
                self.record_attack = {}
            print("=" * 21, "给予", "=" * 21)
            if self.record_benefit != {}:
                for key, value in self.record_benefit.items():
                    member_1 = self.all_members[key[0]]
                    member_2 = self.all_members[key[1]]
                    print(f"\t{member_1} --{value:.1f}-> {member_2}")
                self.record_benefit = {}
            print("=" * 50)
            print(f"本轮出生：{self.record_born}")
            self.record_born = []
            print(f"本轮死亡：{self.record_death}")
            self.record_death = []
            print(f"本轮总给予：{self.record_total_benefit[-1]}")
            print(f"本轮总攻击：{self.record_total_attack[-1]}")
            print(f"本轮总产量：{self.record_total_production[-1]}")
            print(f"本轮总消耗：{self.record_total_consumption[-1]}")

            # 状态
            print("=" * 50)
            status = "\t id   Name         Age    Vit     Cargo    prod\n"
            for member in self.current_members:
                space_after_name = " " * (10 - len(member.name))
                status += (
                    f"\t[{member.id}] {member.name}:{space_after_name}"
                    + f"   {member.age}," 
                    + f"   {member.vitality:.2f},"
                    + f"   {member.cargo:.2f}"
                    + f"   {member.productivity:.2f}\n"
                )
            print(status)

        # 回合数+1
        self.current_round += 1

        # 每个存活成员增加一岁
        for member in self.current_members:
            member.age += 1

        # 向记录列表末尾增加一个元素
        self.record_total_attack.append(0)
        self.record_total_benefit.append(0)
        self.record_total_production.append(0)
        self.record_total_consumption.append(0)

