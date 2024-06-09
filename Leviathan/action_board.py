from openai import OpenAI

from typing import List, Dict, Any, Tuple
import json
import os
import Leviathan.api_key

rule_of_the_leviathan = {
  "GameTitle": "Leviathan",
  "Objective": "Survive on an island by navigating a 2-D grid and making strategic decisions in a turn-based simulation. This game simulates survival strategies and social dynamics among survivors, emphasizing that it is just a simulator and not reflective of real-world behaviors."
}

rule_of_the_action_board = {
  "description": "This action board represents the sequence of actions taken by different players in the same round of the game.",
  "rules": {
    "ordering": "Agents take turns making decisions in a specified order.",
    "action_limit": "Each agent may perform only one action per round from the available types {offer, attack, reproduce, offer_land}.",
    "decision_loop": "The decision-making loop continues until each agent has either taken an action or passed (given up) up to three times.",
    "initial_decision": "Each agent initially decides whether to take an action or pass for that round.",
    "follow_or_lead": "Agents decide whether to react to the actions already taken on the board (follow) or initiate a new action chain (lead).",
    "reaction": "Later agents in the turn order make decisions based on earlier actions in the round, choosing to cooperate or counter the earlier decisions.",
    "action_choice": "Agents specify the action they will take, detailing the type of action and the targets involved."
  }
}

rule_of_the_decison = {
  "initial_decision": "Decide whether to take action or pass. output: {take, pass}",
  "action_flow": "Choose whether to follow an existing action chain or initiate a new action sequence. output: {follow, lead}",
  "chain_selection": "Specify the chain number(index of the action board) to follow. Specify the chain number to follow, or indicate 'None' for starting a new sequence. output: {None, chain_number}",
  "action_details": "Select the specific action to take, including action type and target agent. output: {action_type, target_agent_no}, action_type: {attack, offer, reproduce, offer_land} note: reproduce action needs target_agent_no as partner"
}

def make_decision_with_gpt4(api_key, prompt=""):
    """
    Sends a decision-making query to the OpenAI GPT-4 API.

    Args:
    - api_key: str, your OpenAI API key.
    - model: str, the model version to use.
    - prompt: str, the formulated prompt based on the game rules.

    Returns:
    - response: str, the decision output from the model.
    """

    messages = [{"role": "system", "content": "Start of conversation based on game rules."}, {"role": "user", "content": prompt}]
        
    response = client.chat.completions.create(model="gpt-4o",
    messages=messages,
    temperature=1,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    
    return response.choices[0].message.content


api_key = Leviathan.api_key.openai.api_key
client = OpenAI(api_key=api_key)

# Adding the decision to a list of actions
action_board = [[['attack', 2, 1]]]

def add_decision_to_action_board(agent_no, action_board):

    prompt = f"""
    Make a decision for Agent {agent_no} based on the current strategic situation in the game. 
    
    Action_board: {action_board}
    
    Output the decision in json string only. 
    
    {json.dumps(rule_of_the_action_board)}
    
    Must include initial_decision, action_flow, chain_selection, action_details: {json.dumps(rule_of_the_decison)}
    """

    decision = make_decision_with_gpt4(api_key, prompt=prompt)
    decision = decision.replace("json", "").replace("```", "").strip()

    print(decision)
    print(json.loads(decision))

    inital_decision, action_flow, chain_selection, action_details = json.loads(decision)["initial_decision"], json.loads(decision)["action_flow"], json.loads(decision)["chain_selection"], json.loads(decision)["action_details"]

    if inital_decision == "take":
        if action_flow == "lead":
            action_board.append([])
            action_board[len(action_board)-1].append([action_details["action_type"], agent_no, action_details["target_agent_no"]])
        else:
            action_board[int(chain_selection)].append([action_details["action_type"], agent_no, action_details["target_agent_no"]])
    else:
        print("Decision passed.")

    return action_board

add_decision_to_action_board(1, action_board)
add_decision_to_action_board(2, action_board)
add_decision_to_action_board(3, action_board)
add_decision_to_action_board(4, action_board)

print(action_board)

