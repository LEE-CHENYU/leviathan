# Prompt Refactoring - YAML-Based Prompt Management

## Overview

Prompts have been extracted from Python code into separate YAML files for better maintainability, version control, and ease of editing.

## Structure

```
MetaIsland/
├── prompts/
│   ├── base_prompts.yaml              # Common prompt sections
│   ├── agent_action_templates.yaml    # Action decision templates
│   └── mechanism_templates.yaml       # Mechanism proposal templates
├── prompt_loader.py                   # Prompt loading utility
├── agent_code_decision.py            # Uses prompt loader
└── agent_mechanism_proposal.py       # Uses prompt loader
```

## YAML Files

### 1. base_prompts.yaml
Contains common sections used in both action and mechanism prompts:
- `system_attributes` - Available methods and attributes
- `critical_constraints` - Safety checks and constraints
- `communication_section` - Message passing guide
- `game_mechanics` - Core game parameters
- `implementation_patterns` - Best practices
- `error_prevention` - Error handling guide
- `survival_metrics` - Performance metrics
- `challenge_questions` - Strategic thinking prompts
- `final_instruction_action` - Final instruction for actions
- `final_instruction_mechanism` - Final instruction for mechanisms

### 2. agent_action_templates.yaml
Contains templates for agent actions:
- `introduction` - Overview of available systems
- `action_templates` - 7 action templates:
  1. Propose Trade Contract
  2. Sign Pending Contracts
  3. Market Saturation Analysis & Business Pivot
  4. Join/Create Business Partnership
  5. Extract Resources
  6. Place Market Orders
  7. Supply Chain Management
- `economic_strategies` - 5 strategic patterns
- `implementation_guide` - Step-by-step guide

### 3. mechanism_templates.yaml
Contains templates for mechanism proposals:
- `introduction` - Overview of new systems
- `contract_templates` - 4 contract templates:
  1. Resource Definition & Extraction
  2. Simple Trade Contract
  3. Service Contract (Labor/Production)
  4. Business/Partnership
- `physics_templates` - 3 physics templates:
  1. Agricultural Physics
  2. Manufacturing Physics
  3. Market & Pricing Mechanism
- `usage_guidelines` - How to use the systems

## Prompt Loader API

### PromptLoader Class

```python
from MetaIsland.prompt_loader import get_prompt_loader

loader = get_prompt_loader()

# Load base prompts
base = loader.get_base_prompts()

# Load action templates
actions = loader.get_action_templates()

# Load mechanism templates
mechanisms = loader.get_mechanism_templates()

# Format templates into strings
action_text = loader.format_action_templates()
mechanism_text = loader.format_mechanism_templates(member_id=0)

# Build complete prompts
action_prompt = loader.build_action_prompt(
    member_id=0,
    island_ideology="...",
    error_context="...",
    # ... other parameters
)

mechanism_prompt = loader.build_mechanism_prompt(
    member_id=0,
    island_ideology="...",
    error_context="...",
    # ... other parameters
)
```

## Benefits

### 1. **Maintainability**
- Prompts are in structured YAML format
- Easy to read and edit without touching Python code
- Clear separation of concerns

### 2. **Version Control**
- Prompt changes tracked separately from code changes
- Easy to diff prompt modifications
- Can revert prompts independently of code

### 3. **Flexibility**
- Can swap out different prompt sets for experiments
- Easy to A/B test different phrasings
- Can maintain multiple prompt versions

### 4. **Collaboration**
- Non-programmers can edit prompts
- Clear structure makes it easy to find specific sections
- Comments and descriptions in YAML are self-documenting

### 5. **Testing**
- Can test prompts without running full simulation
- Easy to validate prompt structure
- Can generate prompts programmatically for analysis

## Migration

### Before (Hardcoded in Python)
```python
prompt = f"""
    [Section 1]
    Some text here...
    {variable}

    [Section 2]
    More text...
"""
```

### After (YAML + Loader)
```yaml
# prompts/example.yaml
section1: |
  Some text here...

section2: |
  More text...
```

```python
# Python code
loader = get_prompt_loader()
prompts = loader.get_base_prompts()
prompt = f"{prompts['section1']}\n{variable}\n\n{prompts['section2']}"
```

## Usage in Agent Code

### agent_code_decision.py
```python
from MetaIsland.prompt_loader import get_prompt_loader

async def _agent_code_decision(self, member_id):
    loader = get_prompt_loader()

    # Prepare data
    data = self.prepare_agent_data(member_id)

    # Build prompt using loader
    final_prompt = loader.build_action_prompt(
        member_id=member_id,
        island_ideology=self.island_ideology,
        error_context=data['error_context'],
        current_mechanisms=data['current_mechanisms'],
        # ... etc
    )

    # Add challenge questions
    base = loader.get_base_prompts()
    final_prompt += f"\n\n{base['challenge_questions']}"

    # Add final instruction
    final_prompt += f"\n\n{base['final_instruction_action']}"

    # Call LLM
    completion = client.chat.completions.create(...)
```

### agent_mechanism_proposal.py
Similar pattern for mechanism proposals.

## Editing Prompts

To modify prompts:

1. **Edit YAML file** in `MetaIsland/prompts/`
2. **No code changes needed** - loader automatically picks up changes
3. **Test** by running simulation

Example: To add a new action template:
```yaml
# agent_action_templates.yaml
action_templates:
  - name: "New Action Name"
    description: "What it does"
    code: |
      def agent_action(execution_engine, member_id):
          # Your template code here
          pass
```

## Future Enhancements

1. **Multiple Language Support**
   - Create language-specific YAML files
   - Load prompts based on locale

2. **Dynamic Templates**
   - Use Jinja2 for advanced templating
   - Conditional sections based on game state

3. **Prompt Versioning**
   - Track prompt versions in YAML
   - Load specific versions for reproducibility

4. **Prompt Analytics**
   - Log which prompts lead to best agent performance
   - A/B test different prompt variations

5. **Prompt Optimization**
   - Automatically shorten prompts to save tokens
   - Generate summaries of long sections

## Testing

Test the prompt loader:
```python
from MetaIsland.prompt_loader import PromptLoader

loader = PromptLoader()

# Test loading
base = loader.get_base_prompts()
assert 'system_attributes' in base

actions = loader.get_action_templates()
assert 'action_templates' in actions

# Test formatting
action_text = loader.format_action_templates()
assert 'ACTION TEMPLATES' in action_text

print("✓ All prompt loading tests passed!")
```

## Files Modified

- ✅ Created `MetaIsland/prompts/base_prompts.yaml`
- ✅ Created `MetaIsland/prompts/agent_action_templates.yaml`
- ✅ Created `MetaIsland/prompts/mechanism_templates.yaml`
- ✅ Created `MetaIsland/prompt_loader.py`
- ✅ Created `requirements.txt` with PyYAML dependency
- ⏳ TODO: Update `agent_code_decision.py` to use prompt loader
- ⏳ TODO: Update `agent_mechanism_proposal.py` to use prompt loader

## Next Steps

1. Update Python files to use prompt loader
2. Test full simulation with YAML-based prompts
3. Document any prompt customization patterns
4. Consider adding prompt caching for performance
