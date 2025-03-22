# LLM Integration Guidelines

This document outlines best practices and standards for integrating Large Language Models (LLMs) into the Leviathan project, particularly focusing on the MetaIsland module.

## Supported LLM Providers

The project currently supports the following LLM providers:

| Provider | Models | Use Cases |
|----------|--------|-----------|
| OpenAI   | GPT-3.5, GPT-4 | Agent decisions, mechanism proposals |
| Anthropic | Claude models | Advanced reasoning, code generation |
| Deepseek | Deepseek-Coder, Deepseek-Reasoner | Code generation, analysis |
| AISuite | Various models via unified API | Flexible model routing |

## Model Router

Use the model router system to manage LLM provider selection:

```python
from MetaIsland.model_router import model_router

provider, model_id = model_router("deepseek")
```

The model router centralizes model selection and makes it easier to:
- Switch between different providers
- Apply fallback strategies
- Track usage across the application
- Apply consistent settings

## Prompt Management

### Prompt Organization

- Store prompts in dedicated files (e.g., `prompt.py`)
- Use template strings for dynamic content
- Document the purpose and expected output format for each prompt
- Include examples of expected outputs when possible

### Prompt Structure

All prompts should follow a consistent structure:

1. **System Instructions**: Clear, concise instructions about the task
2. **Context**: Relevant information about the current state
3. **Request**: Specific request or query
4. **Output Format**: Expected format for the response

Example:
```python
def agent_decision_prompt(member, relations, context):
    prompt = f"""
    [System]
    You are an agent in a social simulation making decisions based on relationships and current context.
    
    [Context]
    Agent Information: {member}
    Relationships: {relations}
    Current Environment: {context}
    
    [Request]
    Based on the above information, make a decision about what action to take.
    
    [Output Format]
    Return a JSON object with the following structure:
    {{
        "action": "attack|offer|reproduce|clear|offer_land",
        "target_id": <integer>,
        "reasoning": "<explanation>"
    }}
    """
    return prompt
```

## API Interaction

### API Key Management

- Store API keys in environment variables via `.env` file
- Never hardcode API keys in source code
- Use dotenv for loading environment variables

### Error Handling

Implement robust error handling for API calls:

```python
async def call_llm_with_retries(prompt, provider, model_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=f"{provider}:{model_id}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            else:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
```

### Response Parsing

- Always validate and sanitize LLM responses
- Use structured formats (JSON, YAML) for complex responses
- Implement fallbacks for parsing failures

```python
def parse_agent_decision(response_text):
    try:
        # Try to parse as JSON
        data = json.loads(response_text)
        
        # Validate required fields
        if "action" not in data or "target_id" not in data:
            raise ValueError("Missing required fields in response")
            
        # Validate action type
        valid_actions = ["attack", "offer", "reproduce", "clear", "offer_land"]
        if data["action"] not in valid_actions:
            raise ValueError(f"Invalid action: {data['action']}")
            
        return data
    except json.JSONDecodeError:
        # Fallback: try to extract information using regex
        action_match = re.search(r'"action":\s*"([^"]+)"', response_text)
        target_match = re.search(r'"target_id":\s*(\d+)', response_text)
        
        if action_match and target_match:
            return {
                "action": action_match.group(1),
                "target_id": int(target_match.group(1)),
                "reasoning": "Extracted from malformed response"
            }
        
        # If all parsing fails
        raise ValueError("Could not parse agent decision from LLM response")
```

## Code Generation

The project uses LLMs to generate executable code. Follow these guidelines:

### Safety Measures

- Run generated code in a restricted environment
- Verify code doesn't access sensitive resources
- Use timeouts for code execution
- Implement proper error handling

### Code Quality

- Validate generated code against style guidelines
- Ensure code is well-commented
- Test generated code with sample inputs
- Review generated code for logical errors

### Integration Patterns

Use the following pattern for code generation and execution:

```python
async def generate_and_execute_code(context):
    # 1. Generate code
    code_string = await generate_code(context)
    
    # 2. Validate code (syntax, security)
    validated_code = validate_code(code_string)
    
    # 3. Save code for reference
    save_generated_code(validated_code, context)
    
    # 4. Setup execution environment
    setup_execution_env()
    
    # 5. Execute with timeout and error handling
    try:
        result = execute_code_with_timeout(validated_code, timeout=5)
        return result
    except Exception as e:
        handle_execution_error(e)
        return fallback_behavior()
```

## Performance Considerations

- Use async/await for API calls to prevent blocking
- Implement caching for repetitive queries
- Batch similar requests when possible
- Use lower-tier models for simple tasks
- Track token usage to optimize costs

## Testing LLM Integration

- Create mock responses for testing LLM-dependent functions
- Test error handling with simulated API failures
- Validate parsing logic with various response formats
- Compare outputs across different models for consistency

## Monitoring and Logging

- Log all prompts and responses for debugging
- Track token usage and costs
- Monitor API quotas and rate limits
- Implement alerts for API failures

## Ethics and Bias Considerations

- Review prompts for potential bias
- Validate that generated content is appropriate
- Monitor agent behavior for unexpected patterns
- Document known limitations of LLM-driven behaviors 