from math import ceil
import numpy as np
import random

from sniffio import current_async_library

NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

TACTIC_LIST = ['随机', "平均", "政党", "寡头", "独裁", "福利"]

# Survive
VIT_CONSUME = 20
MIN_PRODUCTIVITY = 20
MAX_PRODUCTIVITY = 30

# Rob
GROUP_SIZE = 10				# 抢劫时目击者+参与者的总数量

# Attack
MIN_COURAGE = 0.5
MAX_COURAGE = 1
MIN_ATTACK = 0.3
MAX_ATTACK = 0.5
SPECTATOR_HELP = 0.2	# 参战加成的比例
LIKE_THRESHOLD = 2  		# A对某人喜欢超过这个值，A不会杀这个人
BE_LIKED_THRESHOLD = 4		# 某人对A的喜欢程度超过这个值，A不会杀这个人

# Like
LIKE_WHEN_ATTACKING = 5		# 对某人主动发起攻击（致死）时，（他对攻击者的）好感度降低值
LIKE_WHEN_HELPING = 2		# 对某人帮助（杀死一个人）时，（他对帮助者的）好感度提升值
# LIKE_WHEN_SEEN_ATTACKING = 1
# 							# 发动攻击被看到时，（其他人对攻击者的）好感度降低值

# Respect
RESPECT_AFTER_KILL = 1				# 杀人后得到respect
RESPECT_AFTER_VICTORY = 1			# 获胜后得到respect

#help
ASSIST_THRESHOLD = 2
SURRENDER_THRESHOLD_VITA = 20
SURRENDER_THRESHOLD_LIKE = 2

# Distribute
INEQUALITY_AVERSION = 0.5 	#分配小于平均值时，好感度下降
REVOLUTION_THRESHOLD_SHARE = INEQUALITY_AVERSION * -1	#可调整数据为分配低于平均值数量
REVOLUTION_THRESHOLD_NUMBER = 0.5	#share_list人数比例达到多少发动革命
PARTY_SHARE = 0.7
FRIEND_THRESHOLD = 1.5 		#好感度与平均水平比例高于此值时，成为寡头成员
CRUELTY = 1 				#独裁模式下，分配额与消耗量的比例

from Member import Members

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

		#self.like = np.random.randint(-LIKE_WHEN_HELPING-1, LIKE_WHEN_HELPING+2, size=(self.counts, self.counts), dtype=int)  	#	第i行代表第i个人 *被* 其他人的喜欢程度
		#self.respect = np.random.randint(-RESPECT_AFTER_VICTORY, RESPECT_AFTER_VICTORY+1, size=(self.counts, self.counts), dtype=int)
		self.like = np.zeros((self.counts, self.counts), dtype=float)
		self.respect = np.zeros((self.counts, self.counts), dtype=float)

		self.leader = None
		self.judge_result = 3

		self.killer_list = []
		self.share_list = []

		self.vitality_list = [member.vitality for member in self.player_list0]

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
			if round == 1:
				self.elect()
			self.print_status()
			self.collect()
			self.print_status()
			self.distribute()
			self.print_status()
			self.justice()
			self.consume()
			self.check()
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
				player.vitality = 0
				print(f"{player.name} 饿死了")

				self.like[player.id, :] = 0
				self.like[:, player.id] = 0
				self.respect[player.id, :] = 0
				self.respect[:, player.id] = 0

				self.player_list.remove(player)
				self.current_counts -= 1

			if player == self.leader:
				self.elect()

	def load(self):
		for player in self.player_list:
			player.load()


	def fight(self, team_A, team_B, A_leader=None, B_leader=None):
		# 先挑选队伍双方，为每个人设置参与度
		# 返回值为两个list，分别为结束时的双方存活的人
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
            
		while continue_fight():
			# 打一轮
			A_eng_list = np.array([member.engagement for member in team_A_alive])
			B_eng_list = np.array([member.engagement for member in team_B_alive])
			A_eng_list = A_eng_list / np.sum(A_eng_list)
			B_eng_list = B_eng_list / np.sum(B_eng_list)

			for member in team_A_alive:
				if np.random.rand() <= member.engagement: # 根据参与度，随机判定是否攻击
					target, attack = member.attack_decision_in_fight(team_B_alive, B_eng_list)
					target.vitality -= attack
					# 好感度调整

					self.like[member.id, target.id] -= attack / 50 * LIKE_WHEN_ATTACKING #&被攻击者好感度调整，需修改好感度减少数值
					for team_member in team_A_alive:
						self.like[member.id, team_member.id] += attack / 50 * team_member.engagement * LIKE_WHEN_HELPING #队友好感度调整，需修改好感度减少数值
					
					# 抢到人头，加repect
					if target.vitality <= 0:
						target.vitality = 0
						self.respect[member.id, :member.id] += RESPECT_AFTER_KILL
						self.respect[member.id, member.id+1:] += RESPECT_AFTER_KILL
						print(f"\t{target.name} 被 {member.name} 杀了")

			for member in team_B_alive:
				if np.random.rand() <= member.engagement:
					target, attack = member.attack_decision_in_fight(team_A_alive, A_eng_list)
					target.vitality -= attack

					self.like[member.id, target.id] -= attack / 50 * LIKE_WHEN_ATTACKING #&需修改好感度减少数值
					for team_member in team_B_alive:
						self.like[member.id, team_member.id] += attack / 50 * team_member.engagement * LIKE_WHEN_HELPING #队友好感度调整，需修改好感度减少数值
					
					# 抢到人头，加repect
					if target.vitality <= 0:
						target.vitality = 0
						self.respect[member.id, :member.id] += RESPECT_AFTER_KILL
						self.respect[member.id, member.id+1:] += RESPECT_AFTER_KILL
						print(f"\t{target.name} 被 {member.name} 杀了")

			# 判断死亡
			for member in team_A_alive:
				if member.vitality <= 0:
					self.like[member.id, :] = 0
					self.like[:, member.id] = 0
					self.respect[member.id, :] = 0
					self.respect[:, member.id] = 0
					team_A_alive.remove(member)
					self.player_list.remove(member)
					self.current_counts -= 1

			for member in team_B_alive:
				if member.vitality <= 0:
					self.like[member.id, :] = 0
					self.like[:, member.id] = 0
					self.respect[member.id, :] = 0
					self.respect[:, member.id] = 0
					team_B_alive.remove(member)
					self.player_list.remove(member)
					self.current_counts -= 1

			# 判断投降（调整engagement）
			for member in team_A_alive:
				if member.vitality < SURRENDER_THRESHOLD_VITA:
					if member.like_calculator(team_A_alive, team_B_alive, self.like) < SURRENDER_THRESHOLD_LIKE:
						member.engagement = 0

		return team_A_alive, team_B_alive


	def collect(self):
		print("-采集-")
		self.load()
		self.rob()

	#偷窃

	def rob(self):
		# 重置killer list
		self.killer_list = []

		# 随机生成一些人群
		groups_for_idx = np.array_split(np.arange(self.current_counts), np.ceil(self.current_counts / GROUP_SIZE))

		groups = []
		for group_for_idx in groups_for_idx:
			group = []
			for member_idx in group_for_idx:
				group.append(self.player_list[member_idx])
			groups.append(group)


		# 统计血量
		self.vitality_list = [member.vitality for member in self.player_list0]

		for member_list in groups:
			killer = None
			victim = None
			for member in member_list:
				for member_2 in member_list:
					if member.id == member_2.id:
						continue
					if member.kill_decision(member_2, self):
						killer = member
						victim = member_2
						break
				if killer is not None:
					break

			# 开打
			if killer is not None and killer is not self.leader and victim is not self.leader:
				team_A = [killer]
				team_B = [victim]
				# 助战
				for member in member_list:
					if member == killer or member == victim:
						continue
					if member.is_leader == False:
						member.assist_decision(self, team_A, team_B, A_leader=killer, B_leader=victim)
				killer.engagement = 1
				victim.engagement = 1

				print(f"{killer.name} {[helper.name for helper in team_A]} 对 {victim.name} {[helper.name for helper in team_B]} 发起战斗")
				team_A_alive, team_B_alive = self.fight(team_A, team_B, A_leader=killer, B_leader=victim)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(killer, victim)
				if self.judge_result == 0:
					# victim 胜利
					self.fight_settle(killer, victim, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[victim.id, :victim.id] += RESPECT_AFTER_VICTORY
					self.respect[victim.id, victim.id:] += RESPECT_AFTER_VICTORY

					if killer.vitality > 0:
						self.respect[killer.id, :killer.id] -= RESPECT_AFTER_VICTORY
						self.respect[killer.id, killer.id:] -= RESPECT_AFTER_VICTORY

				elif self.judge_result == 1:
					# killer 胜利
					self.fight_settle(killer, victim, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[killer.id, :killer.id] += RESPECT_AFTER_VICTORY
					self.respect[killer.id, killer.id:] += RESPECT_AFTER_VICTORY

					if victim.vitality > 0:
						self.respect[victim.id, :victim.id] -= RESPECT_AFTER_VICTORY
						self.respect[victim.id, victim.id:] -= RESPECT_AFTER_VICTORY


				elif self.judge_result == 2:
					# 同时死亡 或 投降
					alive_list = team_A_alive + team_B_alive
					# cargo_share = cargo_pool / len(alive_list)
					# for member in alive_list: 
					# 	member.cargo += cargo_share
					# 	member.eat(cargo_share)


				elif self.judge_result == 3:
					print("不可能的情况：两方均未战死或头像")
					exit(-1)

				if killer.vitality > 0:
					self.killer_list.append(killer)

	def fight_judge(self, killer, victim):
		# 判断输赢
		if (killer.vitality <= 0 or killer.engagement <= 0) \
			and (victim.vitality > 0 and victim.engagement > 0):
			# victim 胜利
			return 0
			
		elif (killer.vitality > 0 and killer.engagement > 0) \
			and (victim.vitality <= 0 or victim.engagement <= 0):
			# killer 胜利
			return 1

		elif (killer.vitality <= 0 or killer.engagement <= 0) \
			and (victim.vitality <= 0 and victim.engagement <= 0):
			# 同时死亡 或 投降
			return 2

		elif (killer.vitality > 0 and killer.engagement > 0) \
			and (victim.vitality > 0 and victim.engagement > 0):
			#不可能的情况：两方均未战死或投降
			return 3

	def fight_settle(self, killer_leader, victim_leader, team_A, team_A_alive, team_B, team_B_alive):
		# 涉及财产时应该修改
		cargo_pool = 0
		property_pool = 0

		#根据输赢进行结算
		if self.judge_result == 0:
			# victim_leader 胜利
			for member in team_A:
				if member.vitality <= 0:
					cargo_pool += member.cargo
					member.cargo = 0
			if victim_leader.vitality > 75:
				victim_leader_consume = 0
			else:
				victim_leader_consume = 75 - victim_leader.vitality
				if cargo_pool < victim_leader_consume:
					victim_leader.vitality += cargo_pool
				else:
					victim_leader.vitality = 75
					cargo_pool -= victim_leader_consume
					cargo_share = cargo_pool / len(team_B_alive)
					for member in team_B_alive: 
						if member.id != victim_leader.id:
							member.cargo += cargo_share
							member.eat(cargo_share)
		
		elif self.judge_result == 1:
					# coup_leader 胜利
					for member in team_B:
						if member.vitality <= 0:
							cargo_pool += member.cargo
							member.cargo = 0
					if killer_leader.vitality > 75:
						killer_leader_consume = 0
					else:
						killer_leader_consume = 75 - killer_leader.vitality
						if cargo_pool < killer_leader_consume:
							killer_leader.vitality += cargo_pool
						else:
							killer_leader.vitality = 75
							cargo_pool -= killer_leader_consume
							cargo_share = cargo_pool / len(team_A_alive)
							for member in team_A_alive: 
								if member.id != killer_leader.id:
									member.cargo += cargo_share
									member.eat(cargo_share)

	def distribute(self):
		print("-分配-")
		cargo_pool = 0
		self.share_list = []
		for i in self.player_list:
			if i not in self.killer_list:
				if i.vitality != 0:
					self.share_list.append(i)
				else:
					print(f"已死之人{i.name}留在了名单上") #&临时处理，如果有死人在player_list中，提示
		for i in self.share_list:
			cargo_pool += i.cargo
			i.cargo = 0
		avg_share = cargo_pool / len(self.share_list)
		if self.leader.tactic == "平均":
			for i in self.share_list:
				i.cargo += avg_share
				cargo_pool -= i.cargo
		if self.leader.tactic == "随机":
			for i in range(len(self.share_list)):
				divider = np.sort(np.random.rand(len(self.share_list) - 1) * cargo_pool)
				cargo_splitted = np.concatenate([[0], divider, [cargo_pool]])
				self.share_list[i].cargo += cargo_splitted[i+1] - cargo_splitted[i]
				cargo_pool -= self.share_list[i].cargo
		if self.leader.tactic == "政党":
			party_number = int(len(self.share_list) / 2) + 1
			# print(self.leader.id)
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			party_member = []
			party_member.append(self.player_list0[self.leader.id]) #将分配者自己加入list
			j = 0
			for i in id_list:
				# print(i)
				if self.player_list0[i] in self.share_list:
					if self.player_list0[i] != self.player_list0[self.leader.id]:
						party_member.append(self.player_list0[i]) 
					j += 1
				if j == party_number:
					break
			for member in self.share_list:
				if member in party_member:
					member.cargo += (cargo_pool * PARTY_SHARE / party_number)
					cargo_pool -= member.cargo 
				else:
					member.cargo += (cargo_pool * (1 - PARTY_SHARE) / (len(self.share_list) - party_number))
					cargo_pool -= member.cargo
		if self.leader.tactic == "寡头":
			party_number = 5
			if len(self.share_list) * 0.5 <= party_number:
				party_number = np.ceil(len(self.share_list) * 0.2)
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			party_member = []
			party_member.append(self.player_list0[self.leader.id]) #将分配者自己加入list
			t = sum(self.like[self.leader.id])/len(self.like[self.leader.id]) * FRIEND_THRESHOLD
			j = 0
			for i in id_list:
				# print(i)
				if self.player_list0[i] in self.share_list:
					if self.like[self.leader.id, i] >= t:
						if self.player_list0[i] != self.player_list0[self.leader.id]:
							party_member.append(self.player_list0[i]) 
					j += 1
				if j == party_number:
					break
			for member in self.share_list:
				if member in party_member:
					member.cargo += (cargo_pool * PARTY_SHARE / party_number)
					cargo_pool -= member.cargo 
				else:
					member.cargo += (cargo_pool * (1 - PARTY_SHARE) / (len(self.share_list) - party_number))
					cargo_pool -= member.cargo
		if self.leader.tactic == "独裁":
			cargo_pool0 = cargo_pool

			current_cruelty = CRUELTY

			if VIT_CONSUME < cargo_pool:
				while (VIT_CONSUME * current_cruelty * (len(self.share_list) - 1) + VIT_CONSUME) / cargo_pool > 1:
					current_cruelty *= 0.8

				for p in self.share_list:
					if self.player_list0[p.id] != self.player_list0[self.leader.id]: 
						p.cargo += VIT_CONSUME * current_cruelty
						cargo_pool -= p.cargo 

			self.player_list0[self.leader.id].cargo += cargo_pool #&独裁者的cargo没有处理
			share_precentage = self.leader.cargo/cargo_pool0 * 100
			print("独裁者将分配池的" + str(round(share_precentage,2)) + "%分给了自己")
		if self.leader.tactic == "福利":
			vitality_sum = 0
			for i in self.share_list:
				vitality_sum += i.vitality
				avg_vitality = vitality_sum/len(self.share_list)
			for i in self.share_list:
				i.cargo += avg_share * avg_vitality/i.vitality #&player_list中有死人
				cargo_pool -= i.cargo
		print("对领袖好感度：")
		for f in self.share_list:
			self.like[self.leader.id, f.id] += (f.cargo - avg_share) * INEQUALITY_AVERSION
			print(f"{round(self.like[self.leader.id, f.id],2)}({round((f.cargo - avg_share) * INEQUALITY_AVERSION,2)})")
		self.like[self.leader.id, self.leader.id] = 0
			
	def candidate(self):
		respect_sum = np.sum(self.respect, 1)
		respect_sum_max = np.max(respect_sum)
		respect_maximum_index = np.array(np.where(respect_sum == respect_sum_max))
		leader_id = np.random.choice(respect_maximum_index[0]) #&相同数值处理
		return leader_id

	def elect(self):
		leader_id = self.candidate()
		if self.leader is not None:
			self.leader.is_leader = False
		
		self.leader = self.player_list0[leader_id]
		self.leader.is_leader = True

	#&justice：包含起义和政变
	def justice(self):
		print("-权力争夺-")
		def revolution_trigger():
			revolutionist = []
			for m in self.share_list:
				if self.like[self.leader.id, m.id] < REVOLUTION_THRESHOLD_SHARE:
					revolutionist.append(m)
			return revolutionist

		def coup_trigger():
			coup_leader_id = self.candidate()
			return coup_leader_id

		revolutionist = revolution_trigger()
		print(len(revolutionist), len(self.share_list) * REVOLUTION_THRESHOLD_NUMBER)
		if len(revolutionist) < len(self.share_list) * REVOLUTION_THRESHOLD_NUMBER:
			coup_leader_id = coup_trigger()
			coup_leader = self.player_list0[coup_leader_id]
			if coup_leader is not None and coup_leader is not self.leader:
				team_A = [coup_leader]
				team_B = [self.leader]
				# 助战
				for member in self.player_list:
					if member == coup_leader or member == self.leader:
						continue
					member.assist_decision(self, team_A, team_B, A_leader=coup_leader, B_leader=self.leader)
				coup_leader.engagement = 1
				self.leader.engagement = 1

				print(f"{coup_leader.name} {[helper.name for helper in team_A]} 对 {self.leader.name} {[helper.name for helper in team_B]} 发动政变")
				team_A_alive, team_B_alive = self.fight(team_A, team_B, A_leader=coup_leader, B_leader=self.leader)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(coup_leader, self.leader)

				if self.judge_result == 0:
					# self.leader 胜利
					self.fight_settle(coup_leader, self.leader, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[self.leader.id, :self.leader.id] += RESPECT_AFTER_VICTORY * 2
					self.respect[self.leader.id, self.leader.id:] += RESPECT_AFTER_VICTORY * 2

					if coup_leader.vitality > 0:
						self.respect[coup_leader.id, :coup_leader.id] -= RESPECT_AFTER_VICTORY * 2
						self.respect[coup_leader.id, coup_leader.id:] -= RESPECT_AFTER_VICTORY * 2

				elif self.judge_result == 1:
					# coup_leader 胜利
					self.fight_settle(coup_leader, self.leader, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[coup_leader.id, :coup_leader.id] += RESPECT_AFTER_VICTORY * 2
					self.respect[coup_leader.id, coup_leader.id:] += RESPECT_AFTER_VICTORY * 2

					if self.leader.vitality > 0:
						self.respect[self.leader.id, :self.leader.id] -= RESPECT_AFTER_VICTORY * 2
						self.respect[self.leader.id, self.leader.id:] -= RESPECT_AFTER_VICTORY * 2


				elif self.judge_result == 2:
					# 同时死亡 或 投降
					alive_list = team_A_alive + team_B_alive
					# cargo_share = cargo_pool / len(alive_list)
					# for member in alive_list: 
					# 	member.cargo += cargo_share
					# 	member.eat(cargo_share)
					self.respect[coup_leader.id, :coup_leader.id] -= RESPECT_AFTER_VICTORY
					self.respect[coup_leader.id, coup_leader.id:] -= RESPECT_AFTER_VICTORY

					self.respect[self.leader.id, :self.leader.id] -= RESPECT_AFTER_VICTORY
					self.respect[self.leader.id, self.leader.id:] -= RESPECT_AFTER_VICTORY

				elif self.judge_result == 3:
					print("不可能的情况：两方均未战死或头像")
					exit(-1)
			else:
				print("政局稳定")
				self.judge_result = 3

			print(self.judge_result)
			if self.judge_result == 1:
				self.leader = coup_leader
				self.leader.is_leader = True
			elif self.judge_result == 2:
				self.elect()

		else:
			share_list_respect = []
			share_respect_sum = np.sum(self.respect, 1)
			for r in revolutionist:
				share_list_respect.append(share_respect_sum[r.id])
				share_respect_sum_max = np.max(share_list_respect)
				share_respect_maximum_index = np.array(np.where(share_respect_sum == share_respect_sum_max))
				revolution_leader_id = np.random.choice(share_respect_maximum_index[0])		#&相同数值处理
			
			revolution_leader = self.player_list0[revolution_leader_id]
			print(f"共有{len(revolutionist)}人发动起义，由{revolution_leader.name}领导")

			if revolution_leader is not None and revolution_leader is not self.leader:
				team_A = [revolution_leader]
				team_B = [self.leader]
				# 助战
				for member in self.player_list:
					if member == revolution_leader or member == self.leader:
						continue
					member.assist_decision(self, team_A, team_B, A_leader=revolution_leader, B_leader=self.leader)
				revolution_leader.engagement = 1
				self.leader.engagement = 1

				print(f"{revolution_leader.name} {[helper.name for helper in revolutionist]} 对 {self.leader.name} {[helper.name for helper in team_B]} 发动起义")
				team_A_alive, team_B_alive = self.fight(revolutionist, team_B, A_leader=revolution_leader, B_leader=self.leader)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(revolution_leader, self.leader)
				if self.judge_result == 0:
					# self.leader 胜利
					self.fight_settle(revolution_leader, self.leader, revolutionist, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[self.leader.id, :self.leader.id] += RESPECT_AFTER_VICTORY * 2
					self.respect[self.leader.id, self.leader.id:] += RESPECT_AFTER_VICTORY * 2

					if revolution_leader.vitality > 0:
						self.respect[revolution_leader.id, :revolution_leader.id] -= RESPECT_AFTER_VICTORY * 2
						self.respect[revolution_leader.id, revolution_leader.id:] -= RESPECT_AFTER_VICTORY * 2

				elif self.judge_result == 1:
					# revolution_leader 胜利
					self.fight_settle(revolution_leader, self.leader, revolutionist, team_A_alive, team_B, team_B_alive)

					# Respect
					self.respect[revolution_leader.id, :revolution_leader.id] += RESPECT_AFTER_VICTORY * 2
					self.respect[revolution_leader.id, revolution_leader.id:] += RESPECT_AFTER_VICTORY * 2

					if self.leader.vitality > 0:
						self.respect[self.leader.id, :self.leader.id] -= RESPECT_AFTER_VICTORY * 2
						self.respect[self.leader.id, self.leader.id:] -= RESPECT_AFTER_VICTORY * 2


				elif self.judge_result == 2:
					# 同时死亡 或 投降
					alive_list = team_A_alive + team_B_alive
					# cargo_share = cargo_pool / len(alive_list)
					# for member in alive_list: 
					# 	member.cargo += cargo_share
					# 	member.eat(cargo_share)
					self.respect[revolution_leader.id, :revolution_leader.id] -= RESPECT_AFTER_VICTORY
					self.respect[revolution_leader.id, revolution_leader.id:] -= RESPECT_AFTER_VICTORY

					self.respect[self.leader.id, :self.leader.id] -= RESPECT_AFTER_VICTORY
					self.respect[self.leader.id, self.leader.id:] -= RESPECT_AFTER_VICTORY


				elif self.judge_result == 3:
					print("不可能的情况：两方均未战死或头像")
					exit(-1)

			print(self.judge_result)
			if self.judge_result == 1:
				self.leader = revolution_leader
				self.leader.is_leader = True
			elif self.judge_result == 2:
				self.elect()

	def check(self):
		print("-回合结束-")
		# 每个角色check
		for player in self.player_list:
			player.check(self)

		# 自身好感度、威望为0
		for i in range(self.counts):
			# assert self.like[i, i] == 0
			# assert self.respect[i, i] == 0
			self.like[i, i] = 0
			self.respect[i, i] = 0
		
		if len(self.player_list) <= 3:
			print(f"Last 3 person: {[player.name for player in self.player_list]}")
			print(f"\n"*10)
			exit()

		self.vitality_list = [member.vitality for member in self.player_list0]
	
