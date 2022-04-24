import numpy as np

from Game import LIKE_WHEN_ATTACKING, LIKE_WHEN_HELPING, MIN_ATTACK, MAX_ATTACK, TACTIC_LIST, VIT_CONSUME, MIN_COURAGE, MAX_COURAGE, MIN_PRODUCTIVITY, MAX_PRODUCTIVITY
from Game import LIKE_THRESHOLD, BE_LIKED_THRESHOLD, SPECTATOR_HELP


class Members:
	def __init__(self, name, id, counts):
		self.name = name
		self.id = id
		self.vitality = 50
		self.cargo = 0
		self.property = 0
		self.productivity = int(np.random.rand() * (MAX_PRODUCTIVITY - MIN_PRODUCTIVITY) + MIN_PRODUCTIVITY)
		self.tactic = np.random.choice(TACTIC_LIST)
		self.is_leader = False
		self.engagement = 0			# 仅在参战时使用，每次战斗都不同，需要重新设置
		self.courage = np.random.rand() * (MAX_COURAGE - MIN_COURAGE) + MIN_COURAGE
		# countsList = list(range(counts))
		# del countsList[id] 
		# for i in countsList:
		# 	setattr(self,str('like' + str(i)), 0)
		# 	setattr(self,str('hate' + str(i)), 0)

	def load(self):
		if not self.is_leader:
			self.cargo += self.productivity		#&可以增加随机性

	def consume(self):
		self.vitality -= VIT_CONSUME
		self.eat() #后面改（可能是需要拆成两个func）

	def kill_decision(self, other, game, vit_list):
		# Leader 不杀人、不被杀		
		if self.is_leader or other.is_leader:
			return False

		# 相互喜欢，不杀
		if game.like[other.id, self.id] > LIKE_THRESHOLD:
			return False
		if game.like[self.id, other.id] > BE_LIKED_THRESHOLD:
			return False

		# 好感度不足或血量不足 导致战力不足，不杀
		like_difference = game.like[self.id, :] - game.like[other.id, :]
		like_difference[self.id] = 0
		like_difference[other.id] = 0
		like_difference[np.abs(like_difference) <= 2] = 0	# 小于2不参战
		# 助战方攻击力
		assist_like = np.copy(like_difference)
		assist_like[assist_like < 2 * LIKE_WHEN_HELPING] = 0
		assist_engagement = np.min([assist_like / LIKE_WHEN_ATTACKING, 1])
		assist_attack = np.sum(assist_engagement * vit_list)
		# 敌对方攻击力
		enemy_like = np.copy(like_difference)
		enemy_like[enemy_like > -2 * LIKE_WHEN_HELPING] = 0
		enemy_engagement = np.min([enemy_like / LIKE_WHEN_ATTACKING, 1])
		enemy_attack = np.sum(enemy_engagement * vit_list)
		# 当本队攻击力不足以击杀敌方首领，不杀		
		if (self.vitality + assist_attack) * self.courage < other.vitality * (np.sum(enemy_engagement) + 1):
			return False
		# 当敌方攻击力足以击杀自己，不杀
		if (other.vitality + enemy_attack) > self.vitality * (np.sum(assist_engagement) + 1) * self.courage:
			return False

		return True

	def attack_decision_in_fight(self, attack_list, engagement_list):
		# 判断攻击目标，产生攻击数值
		# 根据参与度，随机选择攻击对象
		target = np.random.choice(attack_list, size=1, p=engagement_list)
		# 计算攻击力，正比于攻击者血量
		attack = (np.random.rand() * (MAX_ATTACK - MIN_ATTACK) + MIN_ATTACK) * self.vitality
		return target[0], attack
		
	def assist_decision(self, game, team_A, team_B, A_leader=None, B_leader=None):
		#&助战决定，side应是member类对象
		if self.is_leader != False:
			return False
		like_difference = game.like[A_leader.id, self.id] - game.like[B_leader.id, self.id] #&需要解决leader为None的情况
		if like_difference > LIKE_WHEN_HELPING:
			if self.vitality < B_leader.vitality: #&根据死亡概率作调整？
				return False
			else:
				team_A.append(self)
				self.engagement = abs(like_difference) / LIKE_WHEN_ATTACKING
		if like_difference < LIKE_WHEN_HELPING * -1:
			if self.vitality < A_leader.vitality:
				return False
			else:
				team_B.append(self)	
				self.engagement = abs(like_difference) / LIKE_WHEN_ATTACKING

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

	def check(self, game):
		if self.vitality > 100:
			print(f"{self.name}'s vitality goes above 100!")
			self.vitality = 100
		if self.vitality <= 0:
			self.vitality = 0
			#game.player_list.remove(self) 	#解决已死之人问题
			for i in range(game.counts):
				game.like[i,self.id] = 0
				game.respect[i,self.id] = 0
				game.like[self.id, i] = 0
				game.respect[self.id, i] = 0
		if self.cargo > 0:
			self.property += self.cargo
			self.cargo = 0 

	def like_calculator(self, team_A_alive, team_B_alive, like):
        #&计算好感度之差判断是否投降,是否战斗过程中对队友好感度必定持续上升？
        # self 在 team_A中
		team_A_like = 0
		team_B_like = 0
		for member in team_A_alive:
			team_A_like += like[member.id, self.id] #对队友/敌人的好感度
		for member in team_A_alive:
			team_B_like += like[member.id, self.id]
		if self in team_A_alive:
			like_difference = team_A_like - team_B_like
		if self in team_B_alive:
			like_difference = team_B_like - team_A_like
		like_difference = like_difference/((len(team_A_alive) + len(team_B_alive)) * 0.5)  #由于like_difference为累计量，需除以人数
		return like_difference

	def distribute(self, Game, tactic):
		# 领导模拟/实现分配，返回参数是更新后的成员和like


	def distribute_decision(self, Game):
		# 领导思考如何分配

		def evaluation():
			# 评估当前决策的好坏
			# 利好群体：
			# 	- 血量不平等程度（标准差）* 公平 EQUALITY
			#	- 小康线以下人数 * 同情 SYMPATHY
			#	- 富裕线以上人数 * 共产	COMMUNIST
			# 利好自身：
			# 	+ 个人血量 * 自私 SELFNESS
			# 	+ 富裕线以上Like增量 * 精英 ELITIST
			# 	+ 小康线一下Like增量 * 群众 POPULIST
			# 	- 对手——respect高的人——血量增量 * 多疑 SUSPICION
			# 	- 预备起义者血量和 / Share_list人数 / REVOLUTION_THRESHOLD_MEMBER_PORTION * 警戒 CAUTION
			#	- （高Like者战斗力 - 低Like战斗力）* 团结 SOLIDARITY  【高Like：Like > LIKE_WHEN_HELPING + Like平均值】