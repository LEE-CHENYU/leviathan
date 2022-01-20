from lib2to3.pgen2.token import NAME
from nis import match
from os import kill
import random
from tkinter.tix import MAX
import numpy as np


NAME_LIST = np.loadtxt("./name_list.txt")[:, 0]
TACTIC_LIST = ['随机', "平均", "政党", "寡头", "独裁"]

# Fight
SPECTATOR_HELP = 0.1
MIN_ATTACK = 0.3
MAX_ATTACK = 0.7

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
		# countsList = list(range(counts))
		# del countsList[id] 
		# for i in countsList:
		# 	setattr(self,str('like' + str(i)), 0)
		# 	setattr(self,str('hate' + str(i)), 0)

	def load(self):
		if not self.is_leader:
			self.cargo += int(random.rand() * self.productivity)

	def consume(self):
		self.vitality -= 25

	def kill_decision(self, other, game):
		if self.is_leader or other.is_leader:
			return False

		killer_bonus = int(np.average([game.like[self.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP) \
			- int(np.average([game.like[other.id, spectator.id] * spectator.vitality for spectator in game.player_list if spectator not in [self, other]]) * SPECTATOR_HELP)
		self_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * self.vitality) + killer_bonus / 2
		other_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * other.vitality) - killer_bonus / 2

		if other_attack >= self.vitality or self_attack < other.vitality:
			return False

		if game.like[other.id, self.id] > 2:
			return False

		if dice() > 3:
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

class Matrix:
	def __init__(self, counts):
		self.mat = np.zeros((counts, counts))
	def up(self, index, value):
		self.mat[index] += value
	def down(self, index, value):
		self.mat[index] -= value


class Game:
	def __init__(self, counts, player_id, player_list0, like, respect):
		self.counts = eval(input("人数: "))
		self.current_counts = counts

		self.player_id = random.randint(0, counts-1)
		self.player_list = [] 				# alive
		for i in range(0, counts):
			self.player_list.append(Members(NAME_LIST[i], i, counts))
			# print(NAME_LIST[i])
		print("\n你是" + self.player_list[player_id].name)

		self.player_list0 = self.player_list # backup array for original player list

		self.like = np.random.randint(-1, 2, size=(counts, counts), dtype=np.int)  	#	第i行代表第i个人 被 其他人的喜欢程度
		self.respect = np.random.randint(-1, 2, size=(counts, counts), dtype=np.int)
		self.leader = None


	def map(self):
		ring = list(range(self.counts))
		ring = sorted(self.player_list, key=lambda k: random.random())
	
		return ring

	def load(self):
		for player in self.player_list:
			player.load()

	def fight(self, killer, victim):
		killer_bonus = int(np.average([self.like[killer.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP) \
			- int(np.average([self.like[victim.id, spectator.id] * spectator.vitality for spectator in self.player_list if spectator not in [killer, victim]]) * SPECTATOR_HELP)
		killer_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * killer.vitality) + killer_bonus / 2
		victim_attack = int((np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * victim.vitality) - killer_bonus / 2

		self.like[killer.id, victim.id] -= 2
		self.like[killer.id, :killer] -= 1
		self.like[killer.id, killer+1:] -= 1

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
			killer.destroy()
			
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
			killer.eat()
			killer.destroy()

		return killer_list

	def collect(self):
		print("-采集-")
		np.random.shuffle(self.player_list)
		print("\n")
		self.load()
		print("\n")
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

			# adj = (i+1) % (len(ring))
			# l, m, r = ring[i-1], ring[i], ring[(adj)]
			# if dice() < 2:
			# 	fight(l, m, ring, info)
			# elif dice() > 2 and dice() < 4:
			# 	fight(m, r, ring, info)
			# elif dice() < 4:
			# 	pass
			# i += 1

	def distribute(self):
		cargo_pool = 0
		for i in player_list:
			cargo_pool += i.cargo
		if self.leader
		print("-分配-")
		
	def elect(self):
		respect_sum = np.sum(self.respect, 1)
		respect_sum_max = np.max(respect_sum)
		respect_maximum_index = np.where(respect_sum == respect_sum_max)
		leader_id = np.random.choice(respect_maximum_index, 1)

		if self.leader is not None:
			self.leader.is_leader = False
		
		self.leader = self.player_list0[leader_id]
		self.leader.is_leader = True

	def distribute():
		

	def check(self):
		for player in self.player_list:
			player.check()


class Info:
	def __init__(self, counts, name_list, player_id, player_list, player_list0, like, respect):
		self.counts = counts
		self.name_list = name_list
		self.player_id = player_id
		self.player_list = player_list
		self.player_list0 = player_list0
		self.like = like
		self.respect = respect

def dice():
	return random.randint(0,5)


def staring(name_list, player_list, player_id, counts):
	for i in range(0, counts):
		player_list.append(Members(name_list[i], i, counts))
		print(name_list[i])
	print("\n你是"+player_list[player_id].name)

	return player_list

def elect(info):
	respect_sum = np.sum(info.respect.mat, 1)
	respect_sum[3] += 2
	respect_sum[5] += 2
	respect_sum_max = np.max(respect_sum)
	respect_maxium_index = np.where(list(respect_sum) == respect_sum_max)
	print(respect_maxium_index)


def index(player):
	name_list = ['Abraham', 'Biden', 'Charles', 'David', 'Elizabeth', 'Franklin', 'George']
	index = name_list.index(player.name)
	
	return index

	player.id

def collect(info):
	print("-采集-")
	ring = map(info)
	print("\n")
	load(info)
	print("\n")
	rob(ring, info)


def map(info):
	ring = list(range(info.counts))
	ring = sorted(info.player_list, key=lambda k: random.random())
	
	return ring

def load(info):
	i = 0
	while i < len(info.player_list):
		m = info.player_list[i]
		m.cargo += random.randint(40, 50)
		print(f"{m.cargo}, {i}")
		i += 1


def rob(ring, info):
	i = 0
	while i < len(ring):
		adj = (i+1) % (len(ring))
		l, m, r = ring[i-1], ring[i], ring[(adj)]
		if dice() < 2:
			fight(l, m, ring, info)
		elif dice() > 2 and dice() < 4:
			fight(m, r, ring, info)
		elif dice() < 4:
			pass
		i += 1


def fight(killer, victim, ring, info):
	killer.vitality += killer.cargo
	victim.vitality += victim.cargo
	kv = killer.vitality
	vv = victim.vitality
	killer.vitality -= vv
	victim.vitality -= kv
	if killer.vitality <= 0 and victim.vitality <= 0: 
		print("犯罪者" + str(killer.name) + "与正当防卫者" + str(victim.name) + "均死亡")
		info.player_list.remove(killer)
		info.player_list.remove(victim)
		ring.remove(killer)
		ring.remove(victim)
		# 注意需要分别从player_list中和ring中移除player
	else:
		if killer.vitality < victim.vitality:
			if killer.vitality <= 0:
				print("犯罪者" + str(killer.name) + " killed by " + str(victim.name))
				info.player_list.remove(killer)
				ring.remove(killer)
				victim.vitality += killer.cargo
				info.respect.up(index(victim), 1)
				print(info.respect.mat, "\n")
		elif killer.vitality > victim.vitality:
			if victim.vitality <= 0:
				print("正当防卫者" + str(victim.name) + " killed by " + str(killer.name))
				info.player_list.remove(victim)
				ring.remove(victim)
				killer.vitality += victim.cargo
				info.respect.up(index(killer), 1)
				print(info.respect.mat, "\n")


def distribute():
	print("-分配-")


def justice():
	print("-权力斗争-")


def end(info):
	print("-回合结束-")
	print("\n")
	print(info.like.mat, "\n\n", info.respect.mat)
	print("\n")
	for i in info.player_list:
		if info.player_list0[info.player_id] not in info.player_list:
			print("You Died!")
			print(str(len(info.player_list))+"人生还")
			print("\n")
			exit()
	if len(info.player_list) == 1:
			print("最后生还者是"+info.player_list[0].name)
			print("\n")
			exit()


def engine():
	counts = eval(input("人数: "))
	player_id = random.randint(0, counts-1)
	player_list = []
	player_list0 = staring(NAME_LIST, player_list, player_id, counts)
	player_list = player_list0[:]
	like = Matrix(counts)
	respect = Matrix(counts)

	# 两个对象的初始化进行封装
	print("\n")
	print(like.mat, "\n\n", respect.mat)
	info = Info(counts, name_list, player_id, player_list, player_list0, like, respect)
	elect(info)

	while True:
		print("\n")
		collect(info)
		print("\n")
		justice()
		print("\n")
		end(info)


engine()