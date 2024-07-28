from typing import List, Dict, Any, Tuple
import json
import Leviathan.api_key

import google.generativeai as genai
import openai

def generate_prompt(
    author: str, 
    record: str,
) -> str:
    
    return f"""Now you are {author} write a historic literature in the style of your time with the following decision and statistics:{record}"""

def generate_story_using_gpt(
    author: str, 
    log_path: str,
    lang: str = "en"  # Added language parameter for multilingual support
) -> Tuple[bool, str]:
    
    with open(f'{log_path}', 'r') as log_file:
        record = log_file.read()

    prompt = generate_prompt(author, record)

    # Add language support to the prompt
    prompt_with_lang = f"Output Language: {lang}\n\n{prompt}"  # Append language information to the prompt

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt_with_lang,
                },
            ],
        )
        output = completion.choices[0].message.content
        
    except Exception as e:
        return False, f"Error: {str(e)}"

    return output