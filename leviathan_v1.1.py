from lib2to3.pgen2.token import NAME
from nis import match
from os import kill
import random
from tkinter.tix import MAX
import numpy as np



NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

# Survive
VIT_CONSUME = 20

# Fight
SPECTATOR_HELP = 0.2	# 参战加成的比例

MIN_ATTACK = 0.3
MAX_ATTACK = 1

LIKE_THRESHOLD = 2  		# A对某人喜欢超过这个值，A不会杀这个人
BE_LIKED_THRESHOLD = 4		# 某人对A的喜欢程度超过这个值，A不会杀这个人

MIN_BRAVITY = 1
MAX_BRAVITY = 1.5

# Distribute
TACTIC_LIST = ['随机', "平均", "政党"]
INEQUALITY_AVERSION = 0.1 #分配小于平均值时，好感度下降
PARTY_SHARE = 0.7

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
		# self.bravity = np.random.rand() * (MAX_BRAVITY - MIN_BRAVITY) + MIN_BRAVITY
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
		self.eat() #后面改

	def kill_decision(self, other, game):
		if self.is_leader or other.is_leader:
			return False
		killer_bonus = int(np.average([game.like[self.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP) \
			- int(np.average([game.like[other.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP)
		self_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * self.vitality) + killer_bonus / 2
		other_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * other.vitality) - killer_bonus / 2

		if other_attack >= self.vitality or self_attack < other.vitality:
			return False

		if game.like[other.id, self.id] > LIKE_THRESHOLD:
			return False
		if game.like[self.id, other.id] > BE_LIKED_THRESHOLD:
			return False

		return True

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

class Matrix:
	def __init__(self, counts):
		self.mat = np.zeros((counts, counts))
	def up(self, index, value):
		self.mat[index] += value
	def down(self, index, value):
		self.mat[index] -= value


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

		self.like = np.random.randint(-1, 2, size=(self.counts, self.counts), dtype=int)  	#	第i行代表第i个人 被 其他人的喜欢程度
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
	
		return ring

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

	def fight(self, killer, victim):
		killer_bonus = int(np.average([self.like[killer.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP) \
			- int(np.average([self.like[victim.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP)
		killer_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * killer.vitality) + killer_bonus / 2
		victim_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * victim.vitality) - killer_bonus / 2

		self.like[killer.id, victim.id] -= 2
		self.like[killer.id, :killer.id] -= 1
		self.like[killer.id, killer.id+1:] -= 1

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
			self.fight(killer_list[i], victim_list[i])

		self.killer_list = killer_list

		
	def elect(self):
		respect_sum = np.sum(self.respect, 1)
		respect_sum_max = np.max(respect_sum)
		respect_maximum_index = np.array(np.where(respect_sum == respect_sum_max))
		leader_id = np.random.choice(respect_maximum_index[0])

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
			j = 0
			for i in id_list:
				# print(i)
				if self.player_list0[i] in share_list:
					party_member.append(self.player_list0[i])
					j += 1
				if j == party_number:
					break
			for member in share_list:
				if member in party_member:
					member.cargo += (cargo_pool * PARTY_SHARE / party_number)
				else:
					member.cargo += (cargo_pool * (1 - PARTY_SHARE) / (len(share_list) - party_number))
		if self.leader.tactic == "寡头":
			pass
		if self.leader.tactic == "独裁":
			pass 
		for i in share_list:
			self.like[self.leader.id, i.id] += (i.cargo - avg_share) * INEQUALITY_AVERSION


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