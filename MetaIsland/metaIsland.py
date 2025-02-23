from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime
import traceback
import os
from collections import defaultdict
import inspect

from MetaIsland.base_island import Island

from MetaIsland.agent_code_decision import _agent_code_decision
from MetaIsland.agent_mechanism_proposal import _agent_mechanism_proposal
from MetaIsland.analyze import _analyze

from dotenv import load_dotenv
load_dotenv()

class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        action_board: List[List[Tuple[str, int, int]]] = None,
        agent_modifications: dict = None
    ):
        # Create directories for saving generated code
        self.code_save_path = os.path.join(save_path, 'generated_code')
        os.makedirs(self.code_save_path, exist_ok=True)
        
        # Create subdirectories for different types of code
        self.agent_code_path = os.path.join(self.code_save_path, 'agent_actions')
        self.mechanism_code_path = os.path.join(self.code_save_path, 'mechanisms')
        os.makedirs(self.agent_code_path, exist_ok=True)
        os.makedirs(self.mechanism_code_path, exist_ok=True)
        
        # Add version tracking
        self._VERSION = "2.1"
        
        # Add base class code storage
        self.base_class_code = self._load_base_class_code()
        
        # Add agent modification tracking
        self.agent_modifications = {
            'pre_init': [],
            'post_init': [],

        }
            
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        self.performance_history = {}  # {member_id: [list_of_performance_metrics]}
        
        # Add code memory tracking
        self.code_memory = {}  # {member_id: [{'code': str, 'performance': float, 'context': dict}]}
        
        # Add execution history tracking
        self.execution_history = {
            'rounds': []
        }

        # Add message storage
        self.messages = {}  # {member_id: [list_of_messages]}

        # Initialize code storage
        self.agent_code_by_member = {}
        
        # Initialize analysis reports
        self.analysis_reports = {}

        self.island_ideology = ""
        
        self.voting_box = {}

    def new_round(self):
        """
        Initialize a new round in the execution history with a structured record.
        """
        # If the parent class has a new_round(), call it.
        if hasattr(super(), "new_round"):
            super().new_round()
            
        round_record = {
            "round_number": len(self.execution_history['rounds']) + 1,
            "timestamp": datetime.now().isoformat(),
            "agent_actions": [],      # List of agent code execution details
            "agent_messages": {},     # Dictionary keyed by member_id
            "mechanism_modifications": {
                "attempts": [],       # Proposed modifications this round
                "executed": []        # Those that have been successfully executed
            },
            "errors": {
                "agent_code_errors": [],
                "mechanism_errors": [],
                "analyze_code_errors": {}
            },
            "relationships": {},       # Relationships logged per member (if needed)
            "generated_code": {}
        }
        self.execution_history['rounds'].append(round_record)
    
    def _load_base_class_code(self) -> dict:
        """Load the source code of base classes as strings."""
        base_code = {}
        
        # Get the source code for Island
        try:
            base_code['base_island'] = inspect.getsource(Island)
        except Exception as e:
            base_code['base_island'] = f"Error loading Island code: {str(e)}"
            
        # Get the source code for Land
        # try:
        #     from MetaIsland.base_land import Land
        #     base_code['base_land'] = inspect.getsource(Land)
        # except Exception as e:
        #     base_code['base_land'] = f"Error loading Land code: {str(e)}"
            
        # # Get the source code for Member
        # try:
        #     from MetaIsland.base_member import Member
        #     base_code['base_member'] = inspect.getsource(Member)
        # except Exception as e:
        #     base_code['base_member'] = f"Error loading Member code: {str(e)}"
            
        self.base_code = base_code

    def offer(self, member_1, member_2):
        super()._offer(member_1, member_2)
        
    def offer_land(self, member_1, member_2):
        super()._offer_land(member_1, member_2)
        
    def attack(self, member_1, member_2):
        super()._attack(member_1, member_2)

    def bear(self, member_1, member_2):
        super()._bear(member_1, member_2)
    
    def expand(self, member_1):
        super()._expand(member_1)

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
                        member_i_id = self.current_members[i].id
                        member_j_id = self.current_members[j].id
                        statement = (f"{rel_map[relation_type]} "
                                     f"(value={val:.2f})")
                        # Replace {i} with actual index+1 (or keep zero-based)
                        # Same for {j}
                        statement = statement.format(i=member_i_id, j=member_j_id)
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

    def clean_code_string(self, code_str: str) -> str:
        """Remove markdown code block markers and clean up the code string."""
        # Remove ```python and ``` markers
        code_str = code_str.replace('```python', '').replace('```', '').strip()
        
        # Remove any leading or trailing whitespace
        lines = code_str.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove any empty lines at start/end while preserving internal empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        return '\n'.join(lines)

    def get_code_memory_summary(self, member_id):
        """Generate a summary of previous code performances for the agent."""
        if member_id not in self.code_memory:
            return "No previous code history."
            
        memory = self.code_memory[member_id]
        if not memory:
            return "No previous code history."
            
        summary = ["Previous code strategies and their outcomes:"]
        
        # Sort memories by performance
        sorted_memories = sorted(memory, key=lambda x: x['performance'], reverse=True)
        
        for i, mem in enumerate(sorted_memories[-5:]):  # Show last 5 memories
            summary.append(f"\nStrategy {i+1} (Performance: {mem['performance']:.2f}):")
            summary.append(f"Context: {mem['context']}")
            summary.append("Code:")
            summary.append(mem['code'])
            if 'error' in mem:
                summary.append(f"Error encountered: {mem['error']}")
                
        return "\n".join(summary)

    def prepare_agent_data(self, member_id):
        """Prepares and returns all necessary data for agent mechanism proposal."""
        member = self.current_members[member_id]
        # Gather relationship info
        relations = self.parse_relationship_matrix(self.relationship_dict)
        features = self.get_current_member_features()

        # Summaries of past code
        code_memory = self.get_code_memory_summary(member_id)

        # Track relationships for logging
        current_round = len(self.execution_history["rounds"])
        self.execution_history['rounds'][current_round-1]['relationships'] = relations

        # Analysis Memory
        analysis_memory = "No previous analysis"
        if member_id in self.analysis_reports and self.analysis_reports[member_id]:
            analysis_list = self.analysis_reports[member_id]
            analysis_memory = f"Previous analysis reports: {analysis_list}"
        
        # Performance Memory
        past_performance = "No previous actions"
        if member_id in self.performance_history and self.performance_history[member_id]:
            perf_list = self.performance_history[member_id]
            avg_perf = sum(perf_list) / len(perf_list)
            past_performance = f"Previous actions resulted in average performance change of {avg_perf:.2f}"

        # Get previous errors for this member
        previous_errors = []
        if (self.execution_history['rounds'] and 
            'errors' in self.execution_history['rounds'][-1] and 
            'mechanism_errors' in self.execution_history['rounds'][-1]['errors']):
            previous_errors = [
                e for e in self.execution_history['rounds'][-1]['errors']['mechanism_errors']
                if e.get('member_id') == member_id
            ]
        
        error_context = "No previous execution errors"
        if previous_errors:
            last_error = previous_errors[-1]
            error_context = (
                f"Last execution error (Round {last_error['round']}):\n"
                f"Error type: {last_error['error']}\n"
                f"Code that caused error:\n{last_error.get('code', '')}"
            )

        # Get any received messages (and clear them)
        received_messages = self.messages.pop(member_id, [])
        message_context = "\n".join(received_messages) if received_messages else "No messages received"

        # Get current game mechanisms and modification attempts
        current_round = len(self.execution_history['rounds'])
        start_round = max(0, current_round - 5)  # Get last 5 rounds or all if less
        
        # Get executed modifications from recent rounds
        current_mechanisms = []
        for round_data in self.execution_history['rounds'][start_round:]:
            current_mechanisms.extend(round_data['mechanism_modifications']['executed'])
            
        # Get modification attempts from recent rounds for this member
        modification_attempts = {}
        for round_data in self.execution_history['rounds'][-5:]:
            round_num = round_data['round_number']
            member_attempts = [
                attempt for attempt in round_data['mechanism_modifications']['attempts']
            ]
            modification_attempts[round_num] = member_attempts

        report = self.analysis_reports[member_id] if member_id in self.analysis_reports else None

        return {
            'member': member,
            'relations': relations,
            'features': features,
            'code_memory': code_memory,
            'analysis_memory': analysis_memory,
            'past_performance': past_performance,
            'error_context': error_context,
            'message_context': message_context,
            'current_mechanisms': current_mechanisms,
            'modification_attempts': modification_attempts,
            'report': report
        }

    ## Prompting Agents
    
    def analyze(self, member_id):
        """Analyze the game state and propose strategic actions."""
        return _analyze(self, member_id)
    
    def agent_code_decision(self, member_id) -> None:
        """Modified to save generated code"""
        agent_code_decision_result = _agent_code_decision(self, member_id)
        if agent_code_decision_result:
            self.save_generated_code(agent_code_decision_result, member_id, 'agent_action')
        return agent_code_decision_result
    
    def agent_mechanism_proposal(self, member_id) -> None:
        """Modified to save generated code"""
        agent_mechanism_proposal_result = _agent_mechanism_proposal(self, member_id)
        if agent_mechanism_proposal_result:
            self.save_generated_code(agent_mechanism_proposal_result, member_id, 'mechanism')
        return agent_mechanism_proposal_result

    def execute_code_actions(self) -> None:
        """Executes all code that the agents wrote (if any) using a restricted namespace."""
        if not hasattr(self, 'agent_code_by_member'):
            self._logger.warning("No agent code to execute.")
            return

        round_num = len(self.execution_history['rounds'])

        for member_id, code_str in self.agent_code_by_member.items():
            if not code_str:
                continue

            print(f"\nExecuting code for Member {member_id}:")
            # print(code_str)

            # Track old stats before executing
            old_survival = self.compute_survival_chance(self.current_members[member_id])
            old_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'survival_chance': old_survival
            }

            error_occurred = None
            try:
                # Track messages for this member
                messages_sent = []
                # Get received messages for this member BEFORE clearing
                received_messages = self.messages.get(member_id, [])
                
                # Modified exec environment with message tracking
                def tracked_send_message(sender, recipient, msg):
                    nonlocal messages_sent
                    messages_sent.append((recipient, msg))
                    self.send_message(sender, recipient, msg)
                    
                local_env = {
                    'execution_engine': self,
                    'send_message': tracked_send_message
                }
                
                # Execute the code in a way that makes the function accessible
                cleaned_code = self.clean_code_string(code_str)
                exec(cleaned_code, local_env)

                if 'agent_action' in local_env and callable(local_env['agent_action']):
                    print(f"Executing agent_action() for Member {member_id}")
                    # Pass self as execution_engine and member_id
                    local_env['agent_action'](self, member_id)
                else:
                    error_occurred = "No valid agent_action() found"
                    print(f"No valid agent_action() found for Member {member_id}")
                    self._logger.warning(f"No valid agent_action() found for member {member_id}.")

            except Exception as e:
                error_occurred = str(e)
                error_info = {
                    'round': round_num,
                    'type': 'agent_code_execution',
                    'member_id': member_id,
                    'code': code_str,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
                print(f"Error executing code for member {member_id}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {member_id}: {e}")

            # Track changes
            new_survival = self.compute_survival_chance(self.current_members[member_id])
            new_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'survival_chance': new_survival
            }
            
            performance_change = new_survival - old_survival

            # Store in code memory
            if member_id not in self.code_memory:
                self.code_memory[member_id] = []
                
            memory_entry = {
                'code': code_str,
                'performance': performance_change,
                'context': {
                    'old_stats': old_stats,
                    'new_stats': new_stats,
                    'round': round_num
                }
            }
            if error_occurred:
                memory_entry['error'] = error_occurred
                
            self.code_memory[member_id].append(memory_entry)

            # Store performance in history
            if member_id not in self.performance_history:
                self.performance_history[member_id] = []
            self.performance_history[member_id].append(performance_change)

            # Log changes for this round
            self.execution_history['rounds'][-1]['agent_actions'].append({
                'member_id': member_id,
                'code_executed': code_str,
                'old_stats': old_stats,
                'new_stats': new_stats,
                'performance_change': performance_change,
                'messages_sent': messages_sent
            })

            # Log messages in round data
            self.execution_history['rounds'][-1]['agent_messages'][member_id] = {
                'received': received_messages,
                'sent': messages_sent
            }
        
        # Save execution history after each round
        self.save_execution_history()
        
        # Clear code after execution
        self.agent_code_by_member = {}

        # Add this line to preserve messages until next decision phase
        self.messages = {k:v for k,v in self.messages.items() if v}  # Only keep non-empty lists

    def save_execution_history(self):
        """Save the execution history to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('execution_histories', exist_ok=True)
        filename = os.path.join('execution_histories', f'execution_history_{timestamp}.json')
        
        try:
            # Convert any non-serializable objects to strings
            history_copy = json.loads(json.dumps(self.execution_history, default=str))
            
            with open(filename, 'w') as f:
                json.dump(history_copy, f, indent=2)
            print(f"\nExecution history saved to {filename}")
        except Exception as e:
            print(f"Error saving execution history: {e}")
            print(traceback.format_exc())

    def print_agent_performance(self):
        """
        Print or log detailed performance metrics for each agent
        including relationship changes and survival probability changes.
        """
        for member_id, performance_list in self.performance_history.items():
            if member_id not in self.current_members:
                continue  # Skip dead members
                
            member = self.current_members[member_id]
            current_survival = self.compute_survival_chance(member)
            avg_perf = sum(performance_list) / len(performance_list) if performance_list else 0
            
            # Get relationship summary
            relations = self.parse_relationship_matrix(self.relationship_dict)
            relation_summary = [r for r in relations if f"member_{member_id}" in r]
            self._logger.info(
                f"\nMember {member_id} Status Report:"
                f"\n- Current survival chance: {current_survival:.2f}"
                f"\n- Performance changes: {performance_list}"
                f"\n- Average performance: {avg_perf:.2f}"
                f"\n- Recent relationships:\n  " + "\n  ".join(relation_summary[-3:])  # Last 3 relationships
            )

    def compute_survival_chance(self, member):
        """
        Compute a more sophisticated survival chance based on:
        - Member's own attributes (vitality, cargo)
        - Relationships with others (beneficial/hostile connections)
        - Current neighborhood situation
        Returns a float representing survival probability
        """
        # Get member's index in current_members list
        member_index = next((i for i, m in enumerate(self.current_members) if m.id == member.id), -1)
        if member_index == -1:
            return 0.0  # Member not found
        
        #  survival from own attributes
        base_survival = member.vitality + member.cargo
        
        # Get relationship bonuses/penalties
        relationship_modifier = 0
        for relation_type, matrix in self.relationship_dict.items():
            if member_index >= matrix.shape[0]:  # This checks ID against matrix size
                continue
            if relation_type == 'benefit':
                # Use nansum to handle potential NaN values
                relationship_modifier += np.nansum(matrix[member_index, :]) * 0.2
            elif relation_type == 'victim':
                # Use nansum to handle potential NaN values
                relationship_modifier -= np.nansum(matrix[member_index, :]) * 0.3
        
        # Neighborhood safety (more neighbors = more risk/opportunity)
        # Add type conversion to ensure numerical operation
        neighbor_count = float(len(member.current_clear_list))
        neighborhood_modifier = -0.1 * neighbor_count
        
        # Ensure final value is finite
        survival_score = base_survival + relationship_modifier + neighborhood_modifier
        return np.nan_to_num(survival_score, nan=50.0)  # Default to 50 if calculation fails

    def send_message(self, sender_id: int, recipient_id: int, message: str):
        """Allow agents to send messages to each other"""
        print(f"[MSG] Member {sender_id} -> Member {recipient_id}: {message!r}")
        # Add validation check
        if any(m.id == recipient_id for m in self.current_members):
            if recipient_id not in self.messages:
                self.messages[recipient_id] = []
            self.messages[recipient_id].append(f"From member_{sender_id}: {message}")
        else:
            print(f"Invalid message recipient {recipient_id} from member {sender_id}")

    def print_agent_messages(self):
        """Print message communication between agents"""
        print("\n=== Agent Message History ===")
        for round_idx, round_data in enumerate(self.execution_history['rounds']):
            print(f"\nRound {round_idx + 1} ({round_data['timestamp']}):")
            if not round_data['agent_messages']:
                print("  No messages exchanged")
                continue
            
            for member_id, comm in round_data['agent_messages'].items():
                print(f"Member {member_id}:")
                if comm['received']:
                    print("  Received:")
                    for msg in comm['received']:
                        print(f"    - {msg}")
                if comm['sent']:
                    print("  Sent:")
                    for recipient, msg in comm['sent']:
                        print(f"    -> Member {recipient}: {msg}")

    def process_voting_mechanism(self):
        """Handle voting process for modification proposals"""
        current_round = len(self.execution_history['rounds'])
        
        # Check each member's votes in their voting box
        for member_id, vote_data in self.voting_box.items():
            if not vote_data['yes_votes']:
                continue
                
            # Add votes to the corresponding modifications
            for mod_index in vote_data['yes_votes']:
                if mod_index >= len(self.execution_history['rounds'][-1]['modification_attempts']):
                    continue
                    
                mod = self.execution_history['rounds'][-1]['modification_attempts'][mod_index]
                
                if mod['ratified'] or 'ratification_condition' not in mod:
                    continue

                # Check if this modification requires voting
                vote_conditions = [c for c in mod['ratification_condition'].get('conditions', []) 
                                  if c['type'] == 'vote']
                
                # Initialize votes dict if needed
                if 'votes' not in mod:
                    mod['votes'] = {'yes': 0}
                
                # Add this member's vote
                mod['votes']['yes'] += 1
                
                # Check if threshold met
                for condition in vote_conditions:
                    total_voters = len(self.current_members)
                    yes_votes = mod['votes']['yes']
                    
                    if yes_votes / total_voters >= condition['threshold']:
                        mod['ratified'] = True
                        mod['ratified_round'] = current_round
                        print(f"Modification from member {mod['member_id']} ratified by vote")
                        
                # Add who voted for what
                if 'voter_records' not in mod:
                    mod['voter_records'] = []
                mod['voter_records'].append({
                    'member_id': member_id,
                    'vote': 'yes',
                    'timestamp': datetime.now().isoformat()
                })
        
        self.voting_box = {}
                    
    def execute_mechanism_modifications(self):
        """Execute ratified modifications"""
        current_round = len(self.execution_history['rounds'])
           
        # Process voting first
        self.process_voting_mechanism()

        # Check remaining ratification conditions
        for mod in self.execution_history['rounds'][-1]['mechanism_modifications']['attempts']:
            if not mod.get('ratified'):
                conditions = mod.get('ratification_condition', {}).get('conditions', [])
                all_conditions_met = True
                
                for condition in conditions:
                    if condition['type'] != 'vote':  # Skip vote conditions handled earlier
                        # Add condition checking logic here
                        all_conditions_met = False
                        break
                        
                if all_conditions_met:
                    mod['ratified'] = True
                    mod['ratified_round'] = current_round
                    self.execution_history['rounds'][-1]['mechanism_modifications']['executed'].append(mod)

        # Execute ratified modifications
        exec_env = {'execution_engine': self}
        
        for mod in self.execution_history['rounds'][-1]['mechanism_modifications']['attempts']:
            if not mod.get('ratified'):
                continue
                
            try:
                print(f"\nExecuting modification code for Member {mod['member_id']}:")
                # print(mod['code'])

                # Execute modification code
                exec(mod['code'], exec_env)
                exec_env['propose_modification'](self, mod['member_id'])
                print(f"Modification code executed successfully for Member {mod['member_id']}")
                
                # Track successful modification
                mod['executed_round'] = current_round
                self.execution_history['rounds'][-1]['mechanism_modifications']['executed'].append(mod)
                
            except Exception as e:
                error_info = {
                    'round': current_round,
                    'type': 'execute_mechanism_modifications', 
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'code': mod['code'],
                    'member_id': mod['member_id']
                }
                self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
                print(f"Error executing code for member {mod['member_id']}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {mod['member_id']}: {e}")

    def save_generated_code(self, code: str, member_id: int, code_type: str) -> None:
        """
        Save generated code to appropriate directory
        
        Args:
            code (str): The code to save
            member_id (int): ID of the member who generated the code
            code_type (str): Type of code ('agent_action' or 'mechanism')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        round_num = len(self.execution_history['rounds'])
        
        if code_type == 'agent_action':
            save_dir = self.agent_code_path
            filename = f'agent_{member_id}_round_{round_num}_{timestamp}.py'
        else:  # mechanism
            save_dir = self.mechanism_code_path
            filename = f'mechanism_{member_id}_round_{round_num}_{timestamp}.py'
            
        file_path = os.path.join(save_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"# Generated code for Member {member_id} in Round {round_num}\n")
                f.write(f"# Generated at: {timestamp}\n\n")
                f.write(self.clean_code_string(code))
            print(f"Saved generated code to {file_path}")
        except Exception as e:
            print(f"Error saving generated code: {e}")
            print(traceback.format_exc())

def main():
    from time import time
    from utils import save
    import os

    rng = np.random.default_rng()
    path = save.datetime_dir("../data")
    exec = IslandExecution(4, (5, 5), path, 2023)
    IslandExecution._RECORD_PERIOD = 1

    round_num = 10
    
    exec.island_ideology = """
    [Island Ideology]
    
    Island is a place of abundant oil. 
    Agents should modify the mechanism to allow them to extract the oil.
    Agents need to the territory to extract oil.
    Agents should build a market place to exchange oil for cargo.
    Agents can choose the quantity and price of oil to sell.
    Agents need to buy oil to expand their territory.
    """
    
    for round_i in range(round_num):
        # Create log file for this round
        header = f"\n{'='*50}\n=== Round {round_i + 1} ===\n{'='*50}"
        print(header)
        
        exec.new_round()
        exec.get_neighbors()
        exec.produce()
        
        print("\nGenerating mechanism modifications...")
        
        for i in range(len(exec.current_members)):
            print(f"Member {i} is deciding...")
            exec.agent_mechanism_proposal(i)
            
        print("\nExecuting mechanism modifications...")
        exec.execute_mechanism_modifications()
        
        print("\nGenerating agent decisions...")
        
        for i in range(len(exec.current_members)):
            print(f"Member {i} is deciding...")
            exec.agent_code_decision(i)
            
        print("\nExecuting agent actions...")
        exec.execute_code_actions()
        exec.consume()
        
        print("\nPerformance Report:")
        
        # Capture performance output
        exec.print_agent_performance()
        exec.print_agent_messages()
        
        # Print status
        exec.log_status(action=True, log_instead_of_print=False)
        
        print(f"\nSurviving members at end of round: {len(exec.current_members)}")
        
        for member in exec.current_members:
            survival_chance = exec.compute_survival_chance(member)
            status = f"Member {member.id}: Vitality={member.vitality:.2f}, Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}"
            print(status)

if __name__ == "__main__":
    main()