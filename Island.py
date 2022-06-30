import numpy as np
from Member import Member

class Island():
    MIN_VICTIM_MEMORY, MAX_VICTIM_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0
    MIN_BENEFIT_MEMORY, MAX_BENEFIT_MEMORY = -50, 100        # 若随机到负值，则该记忆设为0

    def __init__(self, init_member_number) -> None:

        # 初始人数，当前人数
        self.NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

        self.init_member_num = init_member_number
        self.current_member_num = self.init_member_num

        # 初始人物列表，全体人物列表，当前人物列表
        self.init_members = [Member(self.NAME_LIST[i], i) for i in range(self.init_member_num)]
        self.all_members = self._backup_member_list(self.init_members)
        self.current_members = self._backup_member_list(self.init_members)

        # 初始人物关系
        self.victim_memory = np.random.uniform(Island.MIN_VICTIM_MEMORY, Island.MAX_VICTIM_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.victim_memory[self.victim_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.victim_memory, np.nan)

        self.benefit_memory = np.random.uniform(Island.MIN_BENEFIT_MEMORY, Island.MAX_BENEFIT_MEMORY, size=(self.init_member_num, self.init_member_num))
        self.benefit_memory[self.benefit_memory < 0] = 0  # 若随机到负值，则该记忆设为0
        np.fill_diagonal(self.benefit_memory, np.nan)

        self.relationships = [self.victim_memory, self.benefit_memory]



    def _backup_member_list(self, member_list):
        """复制member_list"""
        return [member for member in member_list]

    def _member_list_append(self, append=[], appended_relationship=[]):
        pass

    def _member_list_drop(self, drop=[]):
        drop_id = [member.id for member in drop]
        drop_id = np.sort(drop_id)[::-1]
        for member_id in drop_id:
            del self.current_members[member_id]
            self.current_member_num -= 1

        for idx in range(len(self.relationships)):
            tmp = np.delete(self.relationships[idx], drop_id, axis=0)
            tmp = np.delete(tmp, drop_id, axis=1)
            self.relationships[idx].resize((self.current_member_num, self.current_member_num), refcheck=False)
            self.relationships[idx][:] = tmp
        
        return

    def member_list_modify(self, append=[], drop=[], appended_relationship=[]):
        """修改member_list""" 
        pass

    def relationship_modify(self):
        """保证自身nan"""
        pass

    def save_current_island(self):
        pass

