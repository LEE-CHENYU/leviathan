from nis import match
import random
import numpy as np


class Members:
	def __init__(self, name, id, counts):
		self.name = name
		self.id = id
		self.vitality = 5
		self.cargo = 0
		countsList = list(range(counts))
		del countsList[id] 
		for i in countsList:
			setattr(self,str('like' + str(i)), 0)
			setattr(self,str('hate' + str(i)), 0)
# 		self.hate_list = np.zeros(counts)
	
	# def decide(self, person_2):
	# 	score += (vit1 - vit2) * danxiao
	# 	score += like 
	
	# 	return score


class Matrix:
	def __init__(self, counts):
		self.mat = np.zeros((counts, counts))
	def up(self, index, value):
		self.mat[index] += value
	def down(self, index, value):
		self.mat[index] -= value


# def data(N):	
# 	like_mat = np.zeros((N, N))
# 	respect_mat = np.zeros((N, N))

# 	return like_mat, respect_mat


class Info:
	def __init__(self, counts, name_list, player_id, player_list, player_list0, like, respect):
		self.counts = counts
		self.name_list = name_list
		self.player_id = player_id
		self.player_list = player_list
		self.player_list0 = player_list0
		self.like = like
		self.respect = respect
		# 方便给函数传递信息


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
	counts = eval(input("人数："))
	name_list = ['Abraham', 'Biden', 'Charles', 'David', 'Elizabeth', 'Franklin', 'George']
	player_id = random.randint(0, counts-1)
	player_list = []
	player_list0 = staring(name_list, player_list, player_id, counts)
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