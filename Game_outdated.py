from ast import excepthandler
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time, os

# from sniffio import current_async_library

NAME_LIST = np.random.permutation(np.loadtxt("./name_list.txt", dtype=str))

TACTIC_LIST = ['随机', "平均", "政党", "寡头", "独裁", "福利"]

# Survive
VIT_CONSUME = 20
MIN_PRODUCTIVITY = 20
MAX_PRODUCTIVITY = 30

# Rob
GROUP_SIZE = 10				# 抢劫时目击者+参与者的总数量

# Attack
MIN_COURAGE = 0.5	#默认0.5
MAX_COURAGE = 1		#默认1
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
INEQUALITY_AVERSION = 0.25 	#分配小于平均值时，好感度下降
REVOLUTION_THRESHOLD_LIKE = INEQUALITY_AVERSION * -20	#可调整数据为分配低于平均值数量, 必须小于零
REVOLUTION_THRESHOLD_MEMBER_PORTION = 0.5	#share_list人数比例达到多少发动革命
PARTY_SHARE = 0.6
FRIEND_THRESHOLD = LIKE_THRESHOLD 		# 好感度高于此值时，成为寡头成员
CRUELTY = 1 				# 独裁模式下，分配额与消耗量的比例
WELFARE_DONATION = 0.3 		# 福利模式下，最多从健康人群处扣除多少生活必需品

from Member_outdated import Members

class Game:
	def __init__(self):
		self.counts = eval(input("人数: "))
		self.current_counts = self.counts

		self.player_id = random.randint(0, self.counts-1)
		self.member_list = [] 				# alive
		for i in range(0, self.counts):
			self.member_list.append(Members(NAME_LIST[i], i, self.counts))
			# print(NAME_LIST[i])
		print("\n你是" + self.member_list[self.player_id].name)

		# 按名字顺序重新排序
		self.member_list = [self.member_list[i] for i in np.argsort([member.name for member in self.member_list])]
		for i in range(self.counts):
			self.member_list[i].id = i
		# 备份初始玩家列表
		self.member_list0 = [element for element in self.member_list] 
		self.name_list = [member.name for member in self.member_list0]
		self.member_color_list = np.random.rand(self.counts, 3) * 0.8 + 0.2

		#self.like = np.random.randint(-LIKE_WHEN_HELPING-1, LIKE_WHEN_HELPING+2, size=(self.counts, self.counts), dtype=int)  	#	第i行代表第i个人 *被* 其他人的喜欢程度
		#self.respect = np.random.randint(-RESPECT_AFTER_VICTORY, RESPECT_AFTER_VICTORY+1, size=(self.counts, self.counts), dtype=int)
		self.like = np.zeros((self.counts, self.counts), dtype=float)
		self.respect = np.zeros((self.counts, self.counts), dtype=float)

		self.leader = None
		self.judge_result = 3

		self.vit_tracker = [[member.vitality for member in self.member_list0]]
		self.leader_tracker = [None]
		self.end_of_round_indicator = [True] # 由于记录的内容可能是回合中间的过程，所以利用一个额外的列表来记录这行数据是否记录在回合末尾

		self.killer_list = []
		self.share_list = []

		self.round = 1


	def print_status(self):
		status = ""
		for player in self.member_list:
			status += f"\n\t[{player.name},\t Vit: {player.vitality:.1f},\t Cargo: {player.cargo:.1f},\t Like: {np.average(self.like[player.id]):.1f},\t Resp: {np.average(self.respect[player.id]):.1f}], " 
		print(f"Current surviver: {status}")
		print(f"Current leader: {self.leader.name}, type: {self.leader.tactic}")

	def track_data(self, end_of_round=False):
		vit_list = []
		for i in range(self.counts):
			member = self.member_list0[i]
			if member.is_leader:
				self.leader_tracker.append(member)
			if member.vitality <= 0:
				vit_list.append(None)
			else:
				vit_list.append(member.vitality)
		self.vit_tracker.append(vit_list)
		self.end_of_round_indicator.append(end_of_round)

	def plot_like_and_respect(self, fig, ax0, ax1):
		cmap = plt.cm.RdYlGn
		fontsize = np.min([12, 150/self.counts])

		# Like
		img0 = ax0.imshow(self.like, vmax=np.max(np.abs(self.like)), vmin=-np.max(np.abs(self.like)), cmap=cmap, zorder=1, alpha=0.85)
		ax0.set_title("Like")
		ax0.set_xticks(range(self.counts), fontsize=fontsize)
		ax0.set_xticklabels(self.name_list, rotation=45, rotation_mode="anchor", horizontalalignment="right", verticalalignment="top", fontsize=fontsize)
		ax0.set_yticks(range(self.counts), fontsize=fontsize)
		ax0.set_yticklabels(self.name_list, fontsize=fontsize)
		# 覆盖死亡人物的位置为 RGB [0.3, 0.3, 0.3]
		rgba = np.zeros((self.counts, self.counts, 4))
		for i in range(self.counts):
			if self.member_list0[i].vitality <= 0:
				rgba[i, :, :3] = 0
				rgba[i, :, 3] = 1
				rgba[:, i, :3] = 0
				rgba[:, i, 3] = 1
		ax0.imshow(rgba, zorder=0)

		# Respect
		img1 = ax1.imshow(self.respect, vmax=np.max(np.abs(self.respect)), vmin=-np.max(np.abs(self.respect)), cmap=cmap, zorder=1, alpha=0.85)
		ax1.set_title("Respect")
		ax1.set_xticks(range(self.counts), fontsize=fontsize)
		ax1.set_xticklabels(self.name_list, rotation=45, rotation_mode="anchor", horizontalalignment="right", verticalalignment="top", fontsize=fontsize)
		ax1.set_yticks(range(self.counts), fontsize=fontsize)
		ax1.set_yticklabels(self.name_list, fontsize=fontsize)
		# 覆盖死亡人物的位置为 RGB [0.3, 0.3, 0.3]
		rgba = np.zeros((self.counts, self.counts, 4))
		for i in range(self.counts):
			if self.member_list0[i].vitality <= 0:
				rgba[i, :, :3] = 0
				rgba[i, :, 3] = 1
				rgba[:, i, :3] = 0
				rgba[:, i, 3] = 1
		ax1.imshow(rgba, zorder=0)
		
		divider = make_axes_locatable(ax0)
		cax0 = divider.append_axes("right", size="5%")
		fig.colorbar(img0, cax=cax0)
		divider = make_axes_locatable(ax1)
		cax1 = divider.append_axes("right", size="5%")
		fig.colorbar(img1, cax=cax1)

	def plot_vit(self, ax):
		# 仅在每轮末尾使用
		fontsize = np.min([12, 150/self.counts])

		fig_length = 10
		while fig_length <= self.round:
			fig_length *= 2

		# plot vitality for everyone at all time
		round_list = np.linspace(0, self.round, len(self.vit_tracker))
		for i in range(self.counts):
			ax.scatter(round_list[self.end_of_round_indicator], np.array(self.vit_tracker)[self.end_of_round_indicator][:, i], color=self.member_color_list[i], s=5)
			ax.plot(round_list, np.array(self.vit_tracker)[:, i], color=self.member_color_list[i], label=self.name_list[i])
		ax.set_xlim(-0.3, fig_length+0.3)
		ax.legend(fontsize=fontsize)

		# plot leaders
		for t in range(len(self.leader_tracker)):
			if self.leader_tracker[t] is not None:
				leader = self.leader_tracker[t]
				ax.scatter([round_list[t]], [self.vit_tracker[t][leader.id]], color=self.member_color_list[leader.id], marker="*")

	def plot_status(self, fig):
		# 仅在每轮末尾使用
		plt.clf()

		ax00 = fig.add_subplot(221)
		ax01 = fig.add_subplot(222)
		self.plot_like_and_respect(fig, ax00, ax01)

		ax1 = fig.add_subplot(212)
		self.plot_vit(ax1)

		plt.tight_layout()
		plt.draw()
		

	def run(self):
		fig = plt.figure(figsize=(8, 8))

		def refresh_on_click(event):

			print("#" * 30 + f"  回合: {self.round}  " + "#" * 30)

			np.random.shuffle(self.member_list)
			if self.round == 1:
				self.elect()
			print(f"Current leader: {self.leader.name}, type: {self.leader.tactic}")

			self.collect()

			self.track_data(end_of_round=False)

			self.distribute()
			self.justice()
			self.consume()

			self.check()
			print(f"Current leader: {self.leader.name}, type: {self.leader.tactic}")

			self.track_data(end_of_round=True)

			self.plot_status(fig)

			self.round += 1

		refresh_on_click(1)

		fig.canvas.mpl_connect('button_press_event', refresh_on_click)
		plt.show()
		plt.draw()


	def map(self):
		ring = list(range(self.counts))
		ring = sorted(self.member_list, key=lambda k: random.random())
	
		return ring #&更复杂的地图

	def consume(self):
		for player in self.member_list:
			player.consume()
			if player.vitality <= 0:
				player.vitality = 0
				print(f"{player.name} 饿死了")

				self.like[player.id, :] = 0
				self.like[:, player.id] = 0
				self.respect[player.id, :] = 0
				self.respect[:, player.id] = 0

				self.member_list.remove(player)
				self.current_counts -= 1

			if player == self.leader:
				self.elect()

	def load(self):
		for player in self.member_list:
			player.load()


	def fight(self, team_A, team_B, A_leader=None, B_leader=None, print_fight_details=False):
		# 先挑选队伍双方，为每个人设置参与度
		# 返回值为两个list，分别为结束时的双方存活的人
		team_A_alive = team_A.copy()
		team_B_alive = team_B.copy()

		def continue_fight():

			# 更新是否投降（调整engagement）
			for member in team_A_alive:
				if member.vitality < SURRENDER_THRESHOLD_VITA and member.engagement != 0:
					if member.like_calculator(team_A_alive, team_B_alive, self.like) < SURRENDER_THRESHOLD_LIKE:
						member.engagement = 0
						print(f"\tA: {member.name}({member.vitality:.1f}) 投降了")

			for member in team_B_alive:
				if member.vitality < SURRENDER_THRESHOLD_VITA and member.engagement != 0:
					if member.like_calculator(team_B_alive, team_A_alive, self.like) < SURRENDER_THRESHOLD_LIKE:
						member.engagement = 0
						print(f"\tB: {member.name}({member.vitality:.1f}) 投降了")
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
			A_eng_list = A_eng_list / np.sum(A_eng_list)
			B_eng_list = np.array([member.engagement for member in team_B_alive])
			B_eng_list = B_eng_list / np.sum(B_eng_list)

			random_attack_list = team_A_alive + team_B_alive
			np.random.shuffle(random_attack_list)

			for member in random_attack_list:

				if member in team_A_alive:
					if team_B_alive == []:
						continue

					if np.random.rand() <= member.engagement: # 根据参与度，随机判定是否攻击
						target, attack = member.attack_decision_in_fight(team_B_alive, B_eng_list)
						if print_fight_details:
							print(f"\t{member.name}({member.vitality:.1f}) --[{attack:.1f}]-> {target.name}({target.vitality:.1f})")
						target.vitality -= attack

						# 好感度调整
						self.like[member.id, target.id] -= attack / 50 * LIKE_WHEN_ATTACKING #&被攻击者好感度调整，需修改好感度减少数值
						for team_member in team_A_alive:
							self.like[member.id, team_member.id] += attack / 50 * team_member.engagement * LIKE_WHEN_HELPING #队友好感度调整，需修改好感度减少数值
						
						# 抢到人头，加repect
						if target.vitality <= 0:
							target.vitality = 0

							for member_i in self.member_list:
								if member != member_i:
									self.respect[member.id, member_i.id] += RESPECT_AFTER_KILL

							print(f"\t{member.name} 杀了 {target.name}")
							
							self.like[target.id, :] = 0
							self.like[:, target.id] = 0
							self.respect[target.id, :] = 0
							self.respect[:, target.id] = 0
							self.member_list.remove(target)
							self.current_counts -= 1
							team_B_alive.remove(target)

							# 如果列表还有人，更新eng_list
							if len(team_B_alive) > 0:
								B_eng_list = np.array([member.engagement for member in team_B_alive])
								B_eng_list = B_eng_list / np.sum(B_eng_list)

				if member in team_B_alive:
					if team_A_alive == []:
						continue

					if np.random.rand() <= member.engagement:
						target, attack = member.attack_decision_in_fight(team_A_alive, A_eng_list)
						if print_fight_details:
							print(f"\t{target.name}({target.vitality:.1f}) <-[{attack:.1f}]-- {member.name}({member.vitality:.1f})")
						target.vitality -= attack

						self.like[member.id, target.id] -= attack / 50 * LIKE_WHEN_ATTACKING #&需修改好感度减少数值
						for team_member in team_B_alive:
							self.like[member.id, team_member.id] += attack / 50 * team_member.engagement * LIKE_WHEN_HELPING #队友好感度调整，需修改好感度减少数值
						
						# 抢到人头，加repect
						if target.vitality <= 0:
							target.vitality = 0

							for member_i in self.member_list:
								if member != member_i:
									self.respect[member.id, member_i.id] += RESPECT_AFTER_KILL

							print(f"\t{target.name} 被 {member.name} 杀了")

							self.like[target.id, :] = 0
							self.like[:, target.id] = 0
							self.respect[target.id, :] = 0
							self.respect[:, target.id] = 0
							self.member_list.remove(target)
							self.current_counts -= 1
							team_A_alive.remove(target)

							# 如果列表还有人，更新eng_list
							if len(team_A_alive) > 0:
								A_eng_list = np.array([member.engagement for member in team_A_alive])
								A_eng_list = A_eng_list / np.sum(A_eng_list)

				if not continue_fight():
					break

		return team_A_alive, team_B_alive


	def collect(self):
		print("-采集-")
		self.load()
		self.rob()

	def update_respect(self, member, value):
		for member_i in self.member_list:
			if member != member_i:
				self.respect[member.id, member_i.id] += value

	#偷窃
	def rob(self):
		# 重置killer list
		self.killer_list = []

		# 随机生成一些人群
		groups_for_idx = np.array_split(np.arange(self.current_counts), np.ceil(self.current_counts / GROUP_SIZE))

		vit_list = [member.vitality for member in self.member_list0]

		group_list = []
		for group_for_idx in groups_for_idx:
			group = []
			for member_idx in group_for_idx:
				group.append(self.member_list[member_idx])
			group_list.append(group)

		for group in group_list:
			killer = None
			victim = None
			for member in group:
				for member_2 in group:
					if member.id == member_2.id:
						continue
					if member.kill_decision(member_2, self, vit_list):
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
				for member in group:
					if member == killer or member == victim:
						continue
					if member.is_leader == False:
						member.assist_decision(self, team_A, team_B, A_leader=killer, B_leader=victim)
				killer.engagement = 1
				victim.engagement = 1

				print(f"{killer.name} {[helper.name for helper in team_A]} 对 {victim.name} {[helper.name for helper in team_B]} 发起战斗")
				team_A_alive, team_B_alive = self.fight(team_A, team_B, A_leader=killer, B_leader=victim, print_fight_details=True)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(killer, victim)
				if self.judge_result == 0:
					# victim 胜利
					self.fight_settle(killer, victim, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(victim, RESPECT_AFTER_VICTORY)

					if killer.vitality > 0:
						self.update_respect(killer, -RESPECT_AFTER_VICTORY)

				elif self.judge_result == 1:
					# killer 胜利
					self.fight_settle(killer, victim, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(killer, RESPECT_AFTER_VICTORY)

					if victim.vitality > 0:
						self.update_respect(victim, -RESPECT_AFTER_VICTORY)



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
			and (victim.vitality <= 0 or victim.engagement <= 0):
			# 同时死亡 或 投降
			return 2

		elif (killer.vitality > 0 and killer.engagement > 0) \
			and (victim.vitality > 0 and victim.engagement > 0):
			#不可能的情况：两方均未战死或投降
			return 3
		
		else:
			print(f"killer_vita: {killer.vitality:.1f}, killer_engage: {killer.engagement}, vic_vita: {victim.vitality:.1f}, vic_engage: {victim.engagement}")

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
		for i in self.member_list:
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

			party_member = [self.leader]
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			i = 0
			while len(party_member) < party_number:
				if self.member_list0[id_list[i]] == self.leader:
					i += 1
					continue
				if self.member_list0[id_list[i]].vitality <= 0:
					i += 1
					continue
				party_member.append(self.member_list0[id_list[i]])
				i += 1
			print("party member: ", [member.name for member in party_member])
			
			party_member_share = (cargo_pool * PARTY_SHARE) / party_number
			opposition_party_member_share = (cargo_pool * (1 - PARTY_SHARE)) / (len(self.share_list) - party_number)
			for member in self.share_list:
				if member in party_member:
					member.cargo += party_member_share
					cargo_pool -= member.cargo 
				else:
					member.cargo += opposition_party_member_share
					cargo_pool -= member.cargo
		if self.leader.tactic == "寡头":
			party_number = 5

			party_member = [self.leader]
			id_list = np.argsort(self.like[self.leader.id])[::-1]
			i = 0

			while len(party_member) < party_number and i < self.counts:
				if self.member_list0[id_list[i]] == self.leader:
					i += 1
					continue
				if self.member_list0[id_list[i]].vitality <= 0:
					i += 1
					continue
				party_member.append(self.member_list0[id_list[i]])
				i += 1

			party_number = len(party_member)
			print("party member: ", [member.name for member in party_member])

			# t = sum(self.like[self.leader.id])/len(self.like[self.leader.id]) * FRIEND_THRESHOLD
			# j = 0
			# for i in id_list:
			# 	# print(i)
			# 	if self.member_list0[i] in self.share_list:
			# 		if self.like[self.leader.id, i] >= t:
			# 			if self.member_list0[i] != self.member_list0[self.leader.id]:
			# 				party_member.append(self.member_list0[i]) 
			# 		j += 1
			# 	if j == party_number:
			# 		break

			############################################################
			#############
			##
			#
			# 修改分配策略：平民维生，团队喂到好感度满，自己拿剩下
			#
			#
			#################

			party_member_share = (cargo_pool * PARTY_SHARE) / party_number
			if len(self.share_list) > party_number:
				opposition_party_member_share = (cargo_pool * (1 - PARTY_SHARE)) / (len(self.share_list) - party_number)
			for member in self.share_list:
				if member in party_member:
					member.cargo += party_member_share
					cargo_pool -= member.cargo 
				else:
					member.cargo += opposition_party_member_share
					cargo_pool -= member.cargo

		if self.leader.tactic == "独裁":
			cargo_pool0 = cargo_pool

			current_cruelty = CRUELTY

			if VIT_CONSUME < cargo_pool:
				while (VIT_CONSUME * current_cruelty * (len(self.share_list) - 1) + VIT_CONSUME) / cargo_pool > 1:
					current_cruelty *= 0.8

				for p in self.share_list:
					if self.member_list0[p.id] != self.member_list0[self.leader.id]: 
						p.cargo += VIT_CONSUME * current_cruelty
						cargo_pool -= p.cargo 

			self.member_list0[self.leader.id].cargo += cargo_pool #&独裁者的cargo没有处理
			share_precentage = self.leader.cargo/cargo_pool0 * 100
			print(f"独裁者将分配池的{share_precentage:.2f}%分给了自己")
		if self.leader.tactic == "福利":
			share_member_vit_list = []
			for i in self.share_list:
				share_member_vit_list.append(i.vitality)
			share_member_vit_list = np.array(share_member_vit_list)
			avg_vitality = np.sum(share_member_vit_list) / len(self.share_list)

			# 先给所有人保底的生活需求
			if VIT_CONSUME * (1 - WELFARE_DONATION) * len(self.share_list) >= cargo_pool:
				# 不够满足保底需求，均分
				basic_share = cargo_pool / len(self.share_list)
				cargo_pool = 0
			else:
				basic_share = (1 - WELFARE_DONATION) * VIT_CONSUME
				cargo_pool -= basic_share * len(self.share_list)
			
			# 剩余的按照血量离满血的远近 加权分配 
			share_member_vit_needed_list = 100 - (share_member_vit_list + basic_share - VIT_CONSUME)
			sum_vit_needed = np.sum(share_member_vit_needed_list)

			for i in range(len(self.share_list)):
				self.share_list[i].cargo += basic_share + cargo_pool * share_member_vit_needed_list[i] / sum_vit_needed
			
			cargo_pool = 0

		# 好感度更新
		# print("对领袖好感度：")
		for f in self.share_list:
			if not f.is_leader:
				self.like[self.leader.id, f.id] += (f.cargo - avg_share) * INEQUALITY_AVERSION
				# print(f"{self.like[self.leader.id, f.id]:.2f} ({(f.cargo - avg_share) * INEQUALITY_AVERSION:.2f})")
		self.like[self.leader.id, self.leader.id] = 0
			
	def candidate(self):
		max_respect_member = []
		max_respect = -np.inf
		for member in self.member_list:
			current_respect = np.sum(self.respect[member.id, :])
			if current_respect > max_respect:
				max_respect_member = [member]
				max_respect = current_respect
			elif current_respect == max_respect:
				max_respect_member.append(member)
		
		return np.random.choice(max_respect_member).id

	def elect(self):
		leader_id = self.candidate()
		if self.leader is not None:
			self.leader.is_leader = False
		
		self.leader = self.member_list0[leader_id]
		self.leader.is_leader = True

	#&justice：包含起义和政变
	def justice(self):
		print("-权力争夺-")
		def revolution_trigger():
			revolutionist = []
			for m in self.share_list:
				if self.like[self.leader.id, m.id] < REVOLUTION_THRESHOLD_LIKE:
					revolutionist.append(m)
			return revolutionist

		def coup_trigger():
			coup_leader_id = self.candidate()
			return coup_leader_id

		revolutionist = revolution_trigger()
		# print(len(revolutionist), len(self.share_list) * REVOLUTION_THRESHOLD_MEMBER_PORTION)
		if len(revolutionist) < len(self.share_list) * REVOLUTION_THRESHOLD_MEMBER_PORTION:
			coup_leader_id = coup_trigger()
			coup_leader = self.member_list0[coup_leader_id]
			if coup_leader is not None and coup_leader is not self.leader:
				team_A = [coup_leader]
				team_B = [self.leader]
				# 助战
				for member in self.member_list:
					if member == coup_leader or member == self.leader:
						continue
					member.assist_decision(self, team_A, team_B, A_leader=coup_leader, B_leader=self.leader)
				coup_leader.engagement = 1
				self.leader.engagement = 1

				print(f"{coup_leader.name} {[helper.name for helper in team_A]} 对 {self.leader.name} {[helper.name for helper in team_B]} 发动政变")
				# print("政变成员情况")
				# for member in team_A:
				# 	print(f"team A: {member.name} ({member.vitality})")
				# for member in team_B:
				# 	print(f"team B: {member.name} ({member.vitality})")
				team_A_alive, team_B_alive = self.fight(team_A, team_B, A_leader=coup_leader, B_leader=self.leader, print_fight_details=True)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(coup_leader, self.leader)

				if self.judge_result == 0:
					# self.leader 胜利
					print("政变失败")
					self.fight_settle(coup_leader, self.leader, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(self.leader, RESPECT_AFTER_VICTORY * 2)

					if coup_leader.vitality > 0:
						self.update_respect(coup_leader, -RESPECT_AFTER_VICTORY * 2)

				elif self.judge_result == 1:
					# coup_leader 胜利
					print("政变成功")
					self.fight_settle(coup_leader, self.leader, team_A, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(coup_leader, RESPECT_AFTER_VICTORY * 2)

					if self.leader.vitality > 0:
						self.update_respect(self.leader, -RESPECT_AFTER_VICTORY * 2)


				elif self.judge_result == 2:
					print("同时死亡 或 投降")
					alive_list = team_A_alive + team_B_alive
					# cargo_share = cargo_pool / len(alive_list)
					# for member in alive_list: 
					# 	member.cargo += cargo_share
					# 	member.eat(cargo_share)
					self.update_respect(coup_leader, -RESPECT_AFTER_VICTORY)

					self.update_respect(self.leader, -RESPECT_AFTER_VICTORY)


				elif self.judge_result == 3:
					print("不可能的情况：两方均未战死或头像")
					exit(-1)
			else:
				print("政局稳定")
				self.judge_result = 3

			# print(self.judge_result)
			if self.judge_result == 1:
				self.leader.is_leader = False
				self.leader = coup_leader
				self.leader.is_leader = True
			elif self.judge_result == 2:
				self.elect()

		else: # 起义
			max_respect_member = []
			max_respect = -np.inf
			for member in revolutionist:
				current_respect = np.sum(self.respect[member.id, :])
				if current_respect > max_respect:
					max_respect_member = [member]
					max_respect = current_respect
				elif current_respect == max_respect:
					max_respect_member.append(member)

			revolution_leader = np.random.choice(max_respect_member)
			print(f"共有{len(revolutionist)}人发动起义，由{revolution_leader.name}领导")

			if revolution_leader is not None and revolution_leader is not self.leader:
				team_A = [revolution_leader]
				team_B = [self.leader]
				# 助战
				for member in self.member_list:
					if member in revolutionist or member == self.leader:
						continue
					member.assist_decision(self, team_A, team_B, A_leader=revolution_leader, B_leader=self.leader)
				revolution_leader.engagement = 1
				self.leader.engagement = 1

				for member in revolutionist:
					if member != revolution_leader:
						member.engagement = self.like[self.leader.id, member.id]/LIKE_WHEN_ATTACKING * -1

				print(f"{revolution_leader.name} {[helper.name for helper in revolutionist]} 对 {self.leader.name} {[helper.name for helper in team_B]} 发动起义")
				team_A_alive, team_B_alive = self.fight(revolutionist, team_B, A_leader=revolution_leader, B_leader=self.leader, print_fight_details=True)

				# 结算
				self.judge_result = 3
				self.judge_result = self.fight_judge(revolution_leader, self.leader)
				if self.judge_result == 0:
					# self.leader 胜利
					print("起义失败")
					self.fight_settle(revolution_leader, self.leader, revolutionist, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(self.leader, RESPECT_AFTER_VICTORY * 2)

					if revolution_leader.vitality > 0:
						self.update_respect(revolution_leader, -RESPECT_AFTER_VICTORY * 2)

				elif self.judge_result == 1:
					# revolution_leader 胜利
					print("起义成功")
					self.fight_settle(revolution_leader, self.leader, revolutionist, team_A_alive, team_B, team_B_alive)

					# Respect
					self.update_respect(revolution_leader, RESPECT_AFTER_VICTORY * 2)

					if self.leader.vitality > 0:
						self.update_respect(self.leader, -RESPECT_AFTER_VICTORY * 2)


				elif self.judge_result == 2:
					# 同时死亡 或 投降
					alive_list = team_A_alive + team_B_alive
					# cargo_share = cargo_pool / len(alive_list)
					# for member in alive_list: 
					# 	member.cargo += cargo_share
					# 	member.eat(cargo_share)
					self.update_respect(revolution_leader, -RESPECT_AFTER_VICTORY)

					self.update_respect(self.leader, -RESPECT_AFTER_VICTORY)


				elif self.judge_result == 3:
					print("不可能的情况：两方均未战死或头像")
					exit(-1)

			print(self.judge_result)
			if self.judge_result == 1:
				self.leader.is_leader = False
				self.leader = revolution_leader
				self.leader.is_leader = True
			elif self.judge_result == 2:
				self.elect()

	def check(self):
		print("-回合结束-")
		# 每个角色check
		for player in self.member_list:
			player.check(self)

		# 自身好感度、威望为0
		for i in range(self.counts):
			# assert self.like[i, i] == 0
			# assert self.respect[i, i] == 0
			self.like[i, i] = 0
			self.respect[i, i] = 0
		
		if len(self.member_list) <= 5:
			print(f"Last 5 person: {[player.name for player in self.member_list]}")
			current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
			dir = f"./Final Figures/{self.counts}_{current_time}/"
			os.mkdir(dir)
			print(f"文件已经存储到 {dir}")
			plt.savefig(f"{dir}Figure_{current_time}.pdf")
			vit_df = pd.DataFrame(self.vit_tracker, columns=self.name_list)
			vit_df.fillna(0)
			vit_df.to_csv(f"{dir}vitality_tracker_{current_time}.csv")
			print(f"\n"*10)
			exit()
		
		self.like[self.like > LIKE_WHEN_ATTACKING] = LIKE_WHEN_ATTACKING
		self.like[self.like < -LIKE_WHEN_ATTACKING] = -LIKE_WHEN_ATTACKING

		self.vitality_list = [member.vitality for member in self.member_list0]
	
