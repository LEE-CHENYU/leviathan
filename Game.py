import numpy as np
import random
from Member import Members

NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))




class Game:
	def __init__(self):
		self.counts = eval(input("人数: "))
		self.current_counts = self.counts

		self.player_id = random.randint(0, self.counts-1)
		self.player_list = [] 				# alive
		for i in range(0, self.counts):
			self.player_list.append(Members(NAME_LIST[i], i, self.counts))
			# print(NAME_LIST[i])
		print("\n你是" + self.player_list[self.player_id].name)

		self.player_list0 = [element for element in self.player_list] # backup array for original player list

		self.like = np.random.randint(-1, 2, size=(self.counts, self.counts), dtype=int)  	#	第i行代表第i个人 *被* 其他人的喜欢程度
		self.respect = np.random.randint(-1, 2, size=(self.counts, self.counts), dtype=int)
		self.leader = None
		self.killer_list = []

	def print_status(self):
		status = ""
		for player in self.player_list:
			status += f"\n\t[{player.name},\t Vit: {player.vitality:.1f},\t Cargo: {player.cargo:.1f},\t Like: {np.average(self.like[player.id]):.1f},\t Resp: {np.average(self.respect[player.id]):.1f}], " 
		print(f"Current surviver: {status}")
		print(f"Current leader: {self.leader.name}, type: {self.leader.tactic}")

	def run(self):
		round = 1
		while True:
			print("#" * 30 + f"  回合: {round}  " + "#" * 30)
			np.random.shuffle(self.player_list)
			self.elect()
			self.print_status()
			self.collect()
			self.print_status()
			self.distribute()
			self.print_status()
			self.consume()
			self.check()
			#self.justice()
			#self.end()

			round += 1


	def map(self):
		ring = list(range(self.counts))
		ring = sorted(self.player_list, key=lambda k: random.random())
	
		return ring #&更复杂的地图

	def consume(self):
		for player in self.player_list:
			player.consume()
			if player.vitality <= 0:
				print(f"{player.name} 饿死了")
				self.player_list.remove(player)
				self.current_counts -= 1

	def load(self):
		for player in self.player_list:
			player.load()

	def fight(self, team_A, team_B, A_leader=None, B_leader=None):
		# 先挑选队伍双方，为每个人设置参与度
		# 返回值为两个list，分别为结束时的双方存活的人
		loser = ""
		team_A_alive = team_A.copy()
		team_B_alive = team_B.copy()
        
		def continue_fight():
			# 返回True来继续战斗
			if A_leader is not None:
				if A_leader.vitality <= 0 or A_leader.engagement <= 0:
					return False
			else:
				all_died = True
				for member in team_A_alive:
					if member.engagement >= 0:
						all_died = False
						break
				if all_died:
					return False

			if B_leader is not None:
				if B_leader.vitality <= 0 or B_leader.engagement <= 0:
					return False
			else:
				all_died = True
				for member in team_B_alive:
					if member.engagement >= 0:
						all_died = False
						break
				if all_died:
					return False

			return True
            
		def like_calculator(self0):
			#&计算好感度之差判断是否投降,是否战斗过程中对队友好感度必定持续上升？
			for member in team_A_alive:
				team_A_like += self.like[member.id, self0.id] #对队友/敌人的好感度
			for member in team_A_alive:
				team_B_like += self.like[member.id, self0.id]
			if self0 in team_A_alive:
				like_difference = team_A_like - team_B_like
			if self0 in team_B_alive:
				like_difference = team_B_like - team_A_like
			like_difference = like_difference/((len(team_A_alive) + len(team_B_alive)) * 0.5)  #由于like_difference为累计量，需除以人数
			return like_difference


		while continue_fight():
			# 打一轮
			A_eng_list = np.array([member.engagement for member in team_A_alive])
			B_eng_list = np.array([member.engagement for member in team_B_alive])
			A_eng_list /= np.sum(A_eng_list)
			B_eng_list /= np.sum(B_eng_list)
			for member in team_A_alive:
				if np.random.rand() <= member.engagement: # 根据参与度，随机判定是否攻击
					target, attack = member.attack_decision(team_B_alive, B_eng_list)
					target.vitality -= attack
					# 好感度调整

					self.like[member.id, target.id] -= attack/10 #&被攻击者好感度调整，需修改好感度减少数值
					for team_member in team_A_alive:
						self.like[member.id, team_member.id] += attack/len(team_A_alive - 1) #队友好感度调整，需修改好感度减少数值

			for member in team_B_alive:
				if np.random.rand() <= member.engagement:
					target, attack = member.attack_decision(team_A_alive, A_eng_list)
					target.vitality -= attack
					self.like[member.id, target.id] -= attack/10 #&需修改好感度减少数值
					for team_member in team_A_alive:
						self.like[member.id, team_member.id] += attack/len(team_A_alive - 1) #队友好感度调整，需修改好感度减少数值

			# 判断死亡
			for member in team_A_alive:
				if member.vitality <= 0:
					del team_A_alive[member]
					del self.player_list[member]
                    self.current_counts -= 1

			for member in team_B_alive:
				if member.vitality <= 0:
					del team_B_alive[member]
					del self.player_list[member]

			# 判断投降（调整engagement）
			for member in team_A_alive:
				if member.vitality < SURRENDER_THRESHOLD_VITA:
					if like_calculator(member) < SURRENDER_THRESHOLD_LIKE:
						member.engagement = 0

        return None

	def fight_old(self, killer, victim):
		killer_bonus = int(np.average([self.like[killer.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP) \
			- int(np.average([self.like[victim.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP)
		killer_attack = int() + killer_bonus / 2
		victim_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * victim.vitality) - killer_bonus / 2
		#&根据好感度决定是否助战

		self.like[killer.id, victim.id] -= 2
		self.like[killer.id, :killer.id] -= 1
		self.like[killer.id, killer.id+1:] -= 1 #好感度调整与犯罪与否有关

		killer.vitality -= victim_attack
		victim.vitality -= killer_attack

		if killer.vitality <= 0 and victim.vitality <= 0: 
			print("犯罪者" + str(killer.name) + "与正当防卫者" + str(victim.name) + "均死亡")
			self.player_list.remove(killer)
			self.player_list.remove(victim)
			self.current_counts -= 2
			# 注意需要分别从player_list中和ring中移除player
		elif killer.vitality > 0 and victim.vitality <= 0:
			print("正当防卫者" + str(victim.name) + " killed by " + str(killer.name))
			self.player_list.remove(victim)

			killer.cargo += victim.cargo
			killer.eat()
			killer.destroy_cargo()
			
			self.respect[killer.id, :] += 1
			self.current_counts -= 1
			# print(self.respect, "\n")
		elif killer.vitality <= 0 and victim.vitality > 0:
			print("犯罪者" + str(killer.name) + " killed by " + str(victim.name))
			self.player_list.remove(killer)

			victim.cargo += killer.cargo
			victim.eat()

			self.respect[victim.id, :] += 2
			self.current_counts -= 1
		else:
			print(f"{killer.name} 和 {victim.name} 交战，但是无人死亡")
			killer.eat()
			killer.destroy_cargo()

	def collect(self):
		print("-采集-")
		self.load()
		self.rob()

	#偷窃

	def rob(self):
		killer_list = []
		victim_list = []
		for i in range(self.current_counts):
			if self.player_list[i].kill_decision(self.player_list[(i+1) % self.current_counts], self):
				if self.player_list[i] not in killer_list and \
					self.player_list[(i+1) % self.current_counts] not in victim_list and \
						self.player_list[i] not in victim_list and \
							self.player_list[(i+1) % self.current_counts] not in killer_list: 
					killer_list.append(self.player_list[i])
					victim_list.append(self.player_list[(i+1) % self.current_counts])
				
			if self.player_list[i].kill_decision(self.player_list[i-1], self):
				if self.player_list[i] not in killer_list and \
					self.player_list[i-1] not in victim_list and \
						self.player_list[i] not in victim_list and \
							self.player_list[(i-1) % self.current_counts] not in killer_list: 
					killer_list.append(self.player_list[i])
					victim_list.append(self.player_list[i-1])
			
		for i in range(len(killer_list)):
			self.fight(killer_list[i], victim_list[i]) #&前面加判定，后面加分配

		self.killer_list = killer_list

		
	def elect(self):
		respect_sum = np.sum(self.respect, 1)
		respect_sum_max = np.max(respect_sum)
		respect_maximum_index = np.array(np.where(respect_sum == respect_sum_max))
		leader_id = np.random.choice(respect_maximum_index[0]) #&相同数值处理

		if self.leader is not None:
			self.leader.is_leader = False
		
		self.leader = self.player_list0[leader_id]
		self.leader.is_leader = True

	def distribute(self):
		print("-分配-")
		cargo_pool = 0
		share_list = []
		for i in self.player_list:
			if i not in self.killer_list:
				share_list.append(i)
		for i in share_list:
			cargo_pool += i.cargo
			i.cargo = 0
		avg_share = cargo_pool / len(share_list)
		if self.leader.tactic == "平均":
			for i in share_list:
				i.cargo += avg_share
				cargo_pool -= i.cargo
		if self.leader.tactic == "随机":
			for i in range(len(share_list)):
				divider = np.sort(np.random.rand(len(share_list) - 1) * cargo_pool)
				cargo_splitted = np.concatenate([[0], divider, [cargo_pool]])
				share_list[i].cargo += cargo_splitted[i+1] - cargo_splitted[i]
				cargo_pool -= share_list[i].cargo
		if self.leader.tactic == "政党":
			party_number = int(len(share_list) / 2) + 1
			# print(self.leader.id)
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			party_member = []
			party_member.append(self.player_list0[self.leader.id]) #将分配者自己加入list
			j = 0
			for i in id_list:
				# print(i)
				if self.player_list0[i] in share_list:
					if self.player_list0[i] != self.player_list0[self.leader.id]:
						party_member.append(self.player_list0[i]) 
					j += 1
				if j == party_number:
					break
			for member in share_list:
				if member in party_member:
					member.cargo += (cargo_pool * PARTY_SHARE / party_number)
					cargo_pool -= member.cargo 
				else:
					member.cargo += (cargo_pool * (1 - PARTY_SHARE) / (len(share_list) - party_number))
					cargo_pool -= member.cargo
		if self.leader.tactic == "寡头":
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			party_member = []
			party_member.append(self.player_list0[self.leader.id]) #将分配者自己加入list
			t = sum(self.like[self.leader.id])/len(self.like[self.leader.id]) * FRIEND_THRESHOLD
			j = 0
			for i in id_list:
				# print(i)
				if self.player_list0[i] in share_list:
					if self.like[self.leader.id, i] >= t:
						if self.player_list0[i] != self.player_list0[self.leader.id]:
							party_member.append(self.player_list0[i]) 
					j += 1
				if j == party_number:
					break
			for member in share_list:
				if member in party_member:
					member.cargo += (cargo_pool * PARTY_SHARE / party_number)
					cargo_pool -= member.cargo 
				else:
					member.cargo += (cargo_pool * (1 - PARTY_SHARE) / (len(share_list) - party_number))
					cargo_pool -= member.cargo
		if self.leader.tactic == "独裁":
			cargo_pool0 = cargo_pool
			for p in share_list:
				if self.player_list0[i] != self.player_list0[self.leader.id]: 
					p.cargo += VIT_CONSUME * CRUELTY
					cargo_pool -= p.cargo 
			self.player_list0[self.leader.id].cargo += cargo_pool
			share_precentage = self.player_list0[self.leader.id].cargo/cargo_pool0 * 100
			print("独裁者将分配池的" + str(share_precentage) + "分给了自己")
		if self.leader.tactic == "福利":
			vitality_sum = 0
			for i in share_list:
				vitality_sum += i.vitality
				avg_vitality = vitality_sum/len(share_list)
			for i in share_list:
				i.cargo += avg_share * avg_vitality/i.vitality
				cargo_pool -= i.cargo
		for i in share_list:
			self.like[self.leader.id, i.id] += (i.cargo - avg_share) * INEQUALITY_AVERSION

	#&justice：包含起义和政变

	def check(self):
		print("-回合结束-")
		for player in self.player_list:
			player.check()
		if len(self.player_list) <= 3:
			print(f"Last 10 person: {[player.name for player in self.player_list]}")
			print(f"\n"*10)
			exit()
	