from ctypes.wintypes import tagRECT
from lib2to3.pgen2.token import NAME
#from nis import match
from os import kill
import random
from tkinter.tix import MAX
from turtle import begin_fill
from urllib.request import AbstractDigestAuthHandler
import numpy as np



NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

# Survive
VIT_CONSUME = 20

# Fight
SPECTATOR_HELP = 0.2	# 参战加成的比例

MIN_ATTACK = 0.3
MAX_ATTACK = 0.5

# Like
LIKE_WHEN_ATTACK = 2		# 对某人主动发起攻击时，（他对攻击者的）好感度降低值
LIKE_WHEN_HELPING = 1		# 对某人帮助时，（他对帮助者的）好感度提升值
LIKE_WHEN_SEEN_ATTACKING = 1
							# 发动攻击被看到时，（其他人对攻击者的）好感度降低值
LIKE_THRESHOLD = 2  		# A对某人喜欢超过这个值，A不会杀这个人
BE_LIKED_THRESHOLD = 4		# 某人对A的喜欢程度超过这个值，A不会杀这个人
# LIKE_BASIS = 1.5			# 好感度对决策的影响：LIKE_BASIS ** like[x, y]

#help
ASSIST_THRESHOLD = 2
SURRENDER_THRESHOLD_VITA = 20
SURRENDER_THRESHOLD_LIKE = 2

MIN_COURAGE = 1
MAX_COURAGE = 1.5

# Distribute
TACTIC_LIST = ['随机', "平均", "政党", "政党", "独裁", "福利"]
INEQUALITY_AVERSION = 0.1 	#分配小于平均值时，好感度下降
PARTY_SHARE = 0.7
FRIEND_THRESHOLD = 1.5 		#好感度与平均水平比例高于此值时，成为寡头成员
CRUELTY = 1.2 				#独裁模式下，分配额与消耗量的比例

def dice():
	return random.randint(0,5)

class Members:
	def __init__(self, name, id, counts):
		self.name = name
		self.id = id
		self.vitality = 50
		self.cargo = 0
		self.productivity = int(np.random.rand() * 20 + 40)
		self.tactic = np.random.choice(TACTIC_LIST)
		self.is_leader = False
		self.engagement = 0			# 仅在参战时使用，每次战斗都不同，需要重新设置
		# self.courage = np.random.rand() * (MAX_COURAGE - MIN_COURAGE) + MIN_COURAGE
		# countsList = list(range(counts))
		# del countsList[id] 
		# for i in countsList:
		# 	setattr(self,str('like' + str(i)), 0)
		# 	setattr(self,str('hate' + str(i)), 0)

	def load(self):
		if not self.is_leader:
			self.cargo += int(np.random.rand() * self.productivity)

	def consume(self):
		self.vitality -= VIT_CONSUME
		self.eat() #后面改（可能是需要拆成两个func）

	def kill_decision(self, other, game):
		killer_bonus = int(np.average([game.like[self.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP) \
			- int(np.average([game.like[other.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP)
		self_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * self.vitality) + killer_bonus / 2
		other_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * other.vitality) - killer_bonus / 2

		if self.is_leader or other.is_leader:
			return False

		if other_attack >= self.vitality or self_attack < other.vitality:
			return False

		if game.like[other.id, self.id] > LIKE_THRESHOLD:
			return False
		if game.like[self.id, other.id] > BE_LIKED_THRESHOLD:
			return False

		return True

	def attack_decision(self, attack_list, engagement_list, like=None):
		# 判断攻击目标，产生攻击数值
		# 根据参与度，随机选择攻击对象
		target = np.random.choice(attack_list, size=1, p=engagement_list)
		# 计算攻击力，正比于攻击者血量
		attack = (np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * self.vitality
		return target, attack
		
	def assist_decision(self, game, team_A, team_B, A_leader=None, B_leader=None):
		#&助战决定，side应是member类对象
		if self.is_leader != False:
			return False
		like_difference = game.like[A_leader.id, self.id] - game.like[B_leader.id, self.id] #&需要解决leader为None的情况
		if like_difference > ASSIST_THRESHOLD:
			if self.vitality < B_leader.vitality: #&根据死亡概率作调整？
				return False
			else:
				team_A.append(self)
				self.engagement = abs(like_difference)/10
		if like_difference < ASSIST_THRESHOLD * -1:
			if self.vitality < A_leader.vitality:
				return False
			else:
				team_B.append(self)	
				self.engagement = abs(like_difference)/10

	def eat(self, amount=None):
		if amount is None:
			if self.vitality + self.cargo >= 100:
				self.cargo -= (100 - self.vitality)
				self.vitality = 100
			else:
				self.vitality += self.cargo
				self.cargo = 0
		else:
			if self.cargo >= amount:
				self.vitality += amount
				self.cargo -= amount
			else:
				self.vitality += self.cargo
				self.cargo = 0

	def destroy_cargo(self):
		self.cargo = 0

	def check(self):
		if self.vitality > 100:
			print(f"{self.name}'s vitality goes above 100!")
			self.vitality = 100

# class Matrix:
# 	def __init__(self, counts):
# 		self.mat = np.zeros((counts, counts))
# 	def up(self, index, value):
# 		self.mat[index] += value
# 	def down(self, index, value):
# 		self.mat[index] -= value


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

			return True

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
					del self.playerlist[member]
					self.current_counts -= 1

			for member in team_B_alive:
				if member.vitality <= 0:
					del team_B_alive[member]
					del self.player_list[member]
					self.current_counts -= 1

			# 判断投降（调整engagement）
			for member in team_A_alive:
				if member.vitality < SURRENDER_THRESHOLD_VITA:
					if like_calculator(member) < SURRENDER_THRESHOLD_LIKE:
						member.engagement = 0

	# def fight_old(self, killer, victim):
	# 	killer_bonus = int(np.average([self.like[killer.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP) \
	# 		- int(np.average([self.like[victim.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP)
	# 	killer_attack = int() + killer_bonus / 2
	# 	victim_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * victim.vitality) - killer_bonus / 2
	# 	#&根据好感度决定是否助战

	# 	self.like[killer.id, victim.id] -= 2
	# 	self.like[killer.id, :killer.id] -= 1
	# 	self.like[killer.id, killer.id+1:] -= 1 #好感度调整与犯罪与否有关

	# 	killer.vitality -= victim_attack
	# 	victim.vitality -= killer_attack

	# 	if killer.vitality <= 0 and victim.vitality <= 0: 
	# 		print("犯罪者" + str(killer.name) + "与正当防卫者" + str(victim.name) + "均死亡")
	# 		self.player_list.remove(killer)
	# 		self.player_list.remove(victim)
	# 		self.current_counts -= 2
	# 		# 注意需要分别从player_list中和ring中移除player
	# 	elif killer.vitality > 0 and victim.vitality <= 0:
	# 		print("正当防卫者" + str(victim.name) + " killed by " + str(killer.name))
	# 		self.player_list.remove(victim)

	# 		killer.cargo += victim.cargo
	# 		killer.eat()
	# 		killer.destroy_cargo()
			
	# 		self.respect[killer.id, :] += 1
	# 		self.current_counts -= 1
	# 		# print(self.respect, "\n")
	# 	elif killer.vitality <= 0 and victim.vitality > 0:
	# 		print("犯罪者" + str(killer.name) + " killed by " + str(victim.name))
	# 		self.player_list.remove(killer)

	# 		victim.cargo += killer.cargo
	# 		victim.eat()

	# 		self.respect[victim.id, :] += 2
	# 		self.current_counts -= 1
	# 	else:
	# 		print(f"{killer.name} 和 {victim.name} 交战，但是无人死亡")
	# 		killer.eat()
	# 		killer.destroy_cargo()

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
	


# def end(info):
# 	print("-回合结束-")
# 	print("\n")
# 	print(info.like.mat, "\n\n", info.respect.mat)
# 	print("\n")
# 	for i in info.player_list:
# 		if info.player_list0[info.player_id] not in info.player_list:
# 			print("You Died!")
# 			print(str(len(info.player_list))+"人生还")
# 			print("\n")
# 			exit()
# 	if len(info.player_list) == 1:
# 			print("最后生还者是"+info.player_list[0].name)
# 			print("\n")
# 			exit()


if __name__ == "__main__":
	engine = Game()
	engine.run()