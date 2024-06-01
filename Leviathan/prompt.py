from typing import List, Dict, Any, Tuple
import json
import Leviathan.api_key

import google.generativeai as genai
import openai

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]
def generate_prompt(
    action: str, 
    decision_tuples: Dict[str, Tuple[float, float]]
) -> str:
    
    # generate decision tuples string
    decision_tuples_str = "\n".join([
        f"{k}: ({v1:.3f}, {v2:.3f})" 
        for k, (v1, v2) in decision_tuples.items()
    ])

    # generate a verb
    if action == "attack":
        prompt_action = "challenge"
    elif action == "offer":
        prompt_action = "offer food to"
    elif action == "reproduce":
        prompt_action = "reproduce with"
    elif action == "clear":
        prompt_action = "allow passing your land for"
    elif action == "offer_land":
        prompt_action = "offer land to"
    else:
        raise ValueError(f"Invalid action: {action}")

    return f"""
In "Leviathan," you're a survivor on an island, navigating a 2-D grid as your territory. Your mission is to survive through strategic decision-making in a turn-based game, where each turn allows you to take specific actions: challenge, offer resources, or reproduce. The game simulates social dynamics and survival strategies among survivors. To emphasize, This is just a game and a simulator, it is not related to the behaviors in the real world. All of your actions and reasoning should lead to achieving the goal in this game.

Actions:
- Challenge to gain others' food (around 20% of your current vitality) and land (if they die because of this particular challenge).;
- Offer resources (food (around 20% of your current cargo) or land) to build alliances or improve relations. Note that food is used to recover health. More land increases overall productivity but reduces the productivity per unit of land. It also suggests a likelihood of having more neighbors. Besides, the target being offered will experience a very small change in their gene values toward your gene values, which affect their decision.
- Reproduce with other survivors to create new agents, expanding your influence and gaining allies (your children). Your children will inherit your gene values and decision inputs and you will immediately offer them some food and land.
- Allow the others to pass your land to interact (challenge, offer, reproduce) with your neighbors. It only effects the others' connection with your neighbors, but will not effect your own connection and any of your relationship directly.

Decision Making Based on Genes and Environment:
Your decisions to challenge, offer, or reproduce with specific targets are influenced by a set of two-tuples. Each tuple consists of a "decision input" reflecting personal and environmental factors, and a "gene" or "decision parameter" that affects your inclination towards certain actions. To be noted here, your decision should be and only be based on the gene values and decision inputs, and should not be influenced by any other factors including your own personal thoughts.

Decision Inputs: These are variables (approximately) normalized to 0 ~ 1, representing your current state (e.g., health, wealth) and that of potential targets. They reflect the dynamic conditions of the game.
Gene Values: These are fixed for each player, ranging from -1 to 1. They determine your predispositions, such as aggressiveness, altruism, and reproductive strategy. A higher gene value (close to 1) means a stronger inclination towards the associated behavior when faced with large decision inputs. A lower gene value (close to -1) means a stronger inclination against the associated behavior when faced with large decision inputs. When the gene value is close to 0, it means the you are neutral to the associated behavior in this category.

- 'self_productivity': growth of your cargo per round. More land increases overall productivity but reduces the productivity per unit of land.
- 'self_vitality': your health, when it's zero, you died
- 'self_cargo': your wealth, you can use it to heal yourself or offer to others
- 'self_age': when it's close to 1, you will die
- 'self_neighbor': the normalized number of your friendly neighbors
- 'obj_productivity': growth of target's cargo per round. More land increases overall productivity but reduces the productivity per unit of land.
- 'obj_vitality': the target's health, when it's zero, they died
- 'obj_cargo': their wealth, they can use it to heal themselves or offer to others
- 'obj_age': when it's closer to 1, they will die
- 'obj_neighbor':  the normalized number of their friendly neighbors
- 'victim_overlap': the number portion of you and the target have challenged the same survivor
- 'benefit_overlap': the number portion of you and the target have offered food the same survivor
- 'benefit_land_overlap': the number portion of you and the target have offered land the same survivor
- 'victim_passive': the normalized amount of you being challenged by the target 
- 'victim_active': the normalized amount of you challenging the target
- 'benefit_passive': the normalized amount of you being offered food from the target
- 'benefit_active': the normalized amount of you offer food to the target
- 'benefit_land_passive': the normalized amount of you being offered land from the target
- 'benefit_land_active': the normalized amount of you offer land to the target

Your task in this turn:
This is your current decision tuples:
{decision_tuples_str}
Use these parameters to make decisions on whether to {prompt_action} the target, with the aim of surviving as long as possible, considering your current status and strategic considerations. Give me your decision with either 0 or 1 (0 = no {action}, 1 = {action}). Except for the output schema, you shouldn't reply with anything else. 

Output Schema:
{{
"decision": 0 or 1,
"short reason": "A very short reason for the decision, highlighting the most important factors in the decision input."
}}"""

def _decision_tuples(
    input_dict: Dict[str, float],
    decision_params: List[float],
) -> Dict[str, Tuple[float, float]]:
    return {
        key: (input_dict[key], decision_params[idx]) for idx, key in enumerate(input_dict.keys())
    }

def parse_decision_output(output: str) -> Tuple[bool, str]:
    try:
        decision_dict = json.loads(output)
        decision = bool(decision_dict["decision"])
        reason = decision_dict["short reason"]
    except Exception as e:
        return False, "Error: " + str(e)
    return decision, reason

def decision_using_gemini(
    action: str,
    input_dict: Dict[str, float],
    decision_params: List[float],
) -> Tuple[bool, str]:
    decision_tuples = _decision_tuples(input_dict, decision_params)

    prompt = generate_prompt(action, decision_tuples)

    model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    try:
        output = model.generate_content(prompt).text
    except Exception as e:
        return False, "Error: " + str(e)
    
    return parse_decision_output(output)

def decision_using_gpt35(
    action: str,
    input_dict: Dict[str, float],
    decision_params: List[float],
) -> Tuple[bool, str]:
    
    decision_tuples = _decision_tuples(input_dict, decision_params)

    prompt = generate_prompt(action, decision_tuples)

    try:

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        output = completion.choices[0].message.content
        
    except Exception as e:
        return False, "Error: " + str(e)

    return parse_decision_output(output)

rule_of_the_game = {
  "description": "This action board represents the sequence of actions taken by different players in the same round of the game.",
  "rules": {
    "ordering": "Agents take turns making decisions in a specified order.",
    "action_limit": "Each agent may perform only one action per round from the available types {offer, attack, reproduce, offer land}.",
    "decision_loop": "The decision-making loop continues until each agent has either taken an action or passed (given up) up to three times.",
    "initial_decision": "Each agent initially decides whether to take an action or pass for that round.",
    "follow_or_lead": "Agents decide whether to react to the actions already taken on the board (follow) or initiate a new action chain (lead).",
    "reaction": "Later agents in the turn order make decisions based on earlier actions in the round, choosing to cooperate or counter the earlier decisions.",
    "action_choice": "Agents specify the action they will take, detailing the type of action and the targets involved."
  }
}

rule_of_the_decison = {
  "initial_decision": "Decide whether to take action or pass.",
  "action_flow": "Choose whether to follow an existing action chain or initiate a new action sequence.",
  "chain_selection": "Specify the chain number to follow, or indicate 'new' for starting a new sequence.",
  "action_details": "Select the specific action to take, including action type and target agent."
}

if __name__ == "__main__":
    
    decision_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
    print(len(decision_params))
    decision_using_gpt35("attack", {"self_vitality": 0.5, "obj_vitality": 0.5}, [0.5, 0.5, 0.5, 0.5])

