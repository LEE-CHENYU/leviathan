from Leviathan.Island import Island
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import openai

class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        action_board: List[List[Tuple[str, int, int]]] = None # change
    ):
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        self.chain_of_action1 = [('attack', 1, 4), ('offer', 2, 1), ('offer', 1, 2)]
        self.chain_of_action2 = [('attack', 4, 1), ('attack', 3, 1)]
        self.chain_of_action3 = [('attack', 1, 4), ('offer', 2, 1), ('offer', 1, 2)]

        self.chain_of_action = [self.chain_of_action1, self.chain_of_action2, self.chain_of_action3]
        self.action_board = action_board if action_board is not None else [] # change

    def offer(self, member_1, member_2, parameter_influence):
        super()._offer(member_1, member_2, parameter_influence)
        
    def attack(self, member_1, member_2):
        super()._attack(member_1, member_2)

    def bear(self, member_1, member_2):
        super()._bear(member_1, member_2)
        
    def execute(self):
        cnt = 0
        
        while True:
            j = 0
            
            for i in range(len(self.action_board)):
                if len(self.action_board[i]) > cnt:
                    chain = self.action_board[i][cnt]
                    dead_list = [member.id for member in self.record_death]
                    if chain[1] in dead_list or chain[2] in dead_list: 
                        continue
                    else:
                        if chain[0] == "attack":
                            self.attack(self.all_members[chain[1]], self.all_members[chain[2]])
                        elif chain[0] == "offer":
                            self.offer(self.all_members[chain[1]], self.all_members[chain[2]], True)
                        elif chain[0] == "bear":
                            self.bear(self.all_members[chain[1]], self.all_members[chain[2]])
                else:
                    j += 1
                    
            cnt += 1
            
            if j >= len(self.action_board):
                break

    def parse_relationship_matrix(self, relationship_dict):
        """
        Parse and return a human-readable summary of the relationship matrices.
        
        :param relationship_dict: A dictionary with keys like 'victim', 'benefit', 'benefit_land'
                                 each containing a NxN numpy array of relationships.
        :return: A list of strings describing the relationships.
        """
        summary = []
        rel_map = {
            'victim':      "member_{i} was attacked by member_{j}",
            'benefit':     "member_{i} gave a benefit to member_{j}",
            'benefit_land':"member_{i} gave land to member_{j}"
        }
        
        for relation_type, matrix in relationship_dict.items():
            if relation_type not in rel_map:
                continue
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    # Filter out invalid or zero entries
                    if not np.isnan(val) and val != 0:
                        # Construct a description
                        statement = (f"{rel_map[relation_type]} "
                                     f"(value={val:.2f})")
                        # Replace {i} with actual index+1 (or keep zero-based)
                        # Same for {j}
                        statement = statement.format(i=i, j=j)
                        summary.append(statement)
        
        return summary
    
    def get_current_member_features(self) -> pd.DataFrame:
        """Collect features for all current members"""
        feature_rows = []
        
        for member in self.current_members:
            # Get self attributes
            feature_row = {
                "self_productivity": member.overall_productivity,
                "self_vitality": member.vitality, 
                "self_cargo": member.cargo,
                "self_age": member.age,
                "self_neighbor": len(member.current_clear_list),
                "member_id": member.id
            }
            feature_rows.append(feature_row)
                
        return pd.DataFrame(feature_rows)
    
    def decision(self, member_id):
        """Get decisions for a member and format them for the action board using GPT"""
        member = self.current_members[member_id]
        relations = self.parse_relationship_matrix(self.relationship_dict)
        features = self.get_current_member_features()
        print(features)
        
        # Check if action board is empty
        action_board_empty = not hasattr(self, 'action_board') or len(self.action_board) == 0
            
        # Create prompt for GPT with just the dataframes
        prompt = f"""
        You are member_{member.id} in a survival game. Your goal is to survive and thrive by making strategic decisions. Consider your relationships with other members and the current state of the action board. Make a decision based on the following information:
        
        Available decision names:
        - 'attack': Challenge another member to take their land
        - 'offer': Give food/resources to another member
        - 'benefit_land': Give land to another member
        
        Features of all members:
        {features}
        
        Relationship matrix between all members:
        {relations}
        
        Current action board status: {"Empty" if action_board_empty else "Has existing actions"}
        
        Output format:
        Line 1 (str): Chain decision
        - "new" to create a new action chain
        - A number(1, 2, 3, ...) to append to that chain
        
        Line 2 (tuple[str, int, int]): Action decision
        - decision_name: One of ['attack', 'offer', 'benefit_land']
        - member_id: Your ID ({member.id})
        - target_id: ID of target member
        
        Example output:
        new
        ('attack', 1, 4)
        
        OR
        
        2
        ('offer', 2, 1)
        
        Output only the formatted decision without explanation or chain of thought.
        """
        
        # Use GPT to make decision
        try:
            completion = openai.chat.completions.create(
                model="o1-mini", 
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = completion.choices[0].message.content.replace('```', '')
        except Exception as e:
            print(e)
            self._logger.error(f"GPT API error: {e}")
            result = 0
        
        return result
    
    def append_to_action_board(self, result: str) -> None:
        """Append the GPT decision result to the action board.
        
        Args:
            result (str): The GPT output containing chain decision and action
        """
        if not result or isinstance(result, int):
            self._logger.warning("Invalid result format, skipping action board update")
            return
            
        try:
            # Split into lines and clean up
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            if len(lines) != 2:
                self._logger.warning(f"Unexpected result format: {result}")
                return
                
            chain_decision = lines[0]
            action_tuple = eval(lines[1])  # Safely evaluate the action tuple string
            
            if not hasattr(self, 'action_board'):
                self.action_board = []
                
            # Create new chain or append to existing
            if chain_decision.lower() == 'new':
                self.action_board.append([action_tuple])
            else:
                try:
                    chain_num = int(chain_decision)
                    if 1 <= chain_num <= len(self.action_board):
                        self.action_board[chain_num-1].append(action_tuple)
                    else:
                        self._logger.warning(f"Invalid chain number: {chain_num}")
                except ValueError:
                    self._logger.warning(f"Invalid chain decision format: {chain_decision}")
                    
        except Exception as e:
            print(e)
            self._logger.error(f"Error appending to action board: {e}")

    
def main():
    
    from Leviathan.Island import Island
    from Leviathan.Member import Member
    from Leviathan.Analyzer import Analyzer
    from time import time
    from Leviathan.Land import Land
    from utils import save
    import os

    rng = np.random.default_rng()
    path = save.datetime_dir("../data")
    exec = IslandExecution(5, (5, 5), path, 2023)
    IslandExecution._RECORD_PERIOD = 1
    Member._DECISION_BACKEND = 'inner product'
    Member._PARAMETER_INFLUENCE = 0

    action_prob = 0.5
    round_num = 10
    
    for i in range(round_num):
        exec.new_round()
        exec.get_neighbors()
        
        for i in range(len(exec.current_members)):
            decision = exec.decision(i)
            print(decision)
            exec.append_to_action_board(decision)
            print(exec.action_board)
        
        exec.execute()
        exec.action_board = []
        exec.log_status(action=True, log_instead_of_print=False)

if __name__ == "__main__":
    main()