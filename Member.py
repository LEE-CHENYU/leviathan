import numpy as np

MIN_ATTACK = 0.3
MAX_ATTACK = 0.5

SPECTATOR_HELP = 0.2	# 参战加成的比例

TACTIC_LIST = ['随机', "平均", "政党", "政党", "独裁", "福利"]

LIKE_THRESHOLD = 2  		# A对某人喜欢超过这个值，A不会杀这个人
BE_LIKED_THRESHOLD = 4		# 某人对A的喜欢程度超过这个值，A不会杀这个人

VIT_CONSUME = 20

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