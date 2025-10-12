# Integration Guide - Using YAML-Based Prompts

## Quick Integration

### In agent_code_decision.py

Replace the hardcoded prompt sections with:

```python
from MetaIsland.prompt_loader import get_prompt_loader

async def _agent_code_decision(self, member_id) -> None:
    # Get prompt loader
    loader = get_prompt_loader()
    base = loader.get_base_prompts()

    # Prepare data (existing code)
    data = self.prepare_agent_data(member_id)

    # Build main prompt using loader
    main_prompt = loader.build_action_prompt(
        member_id=member_id,
        island_ideology=self.island_ideology,
        error_context=data['error_context'],
        current_mechanisms=str(data['current_mechanisms']),
        report=str(data['report']) if data['report'] else "No analysis available",
        features=data['features'].to_string(),
        relations="\n".join(data['relations']),
        code_memory=data['code_memory'],
        past_performance=data['past_performance'],
        analysis_memory=data['analysis_memory'],
        message_context=data['message_context'],
        base_code=str(self.base_class_code)
    )

    # Add challenge questions
    main_prompt += f"\n\n{base['challenge_questions']}"

    # Add final instruction
    final_prompt = main_prompt + f"\n\n{base['final_instruction_action']}"

    # Call LLM (existing code)
    completion = client.chat.completions.create(
        model=f'{provider}:{model_id}',
        messages=[{"role": "user", "content": final_prompt}]
    )
    # ... rest of existing code
```

### In agent_mechanism_proposal.py

Similar integration:

```python
from MetaIsland.prompt_loader import get_prompt_loader

async def _agent_mechanism_proposal(self, member_id) -> None:
    # Get prompt loader
    loader = get_prompt_loader()
    base = loader.get_base_prompts()

    # Prepare data (existing code)
    data = self.prepare_agent_data(member_id)

    # Build main prompt using loader
    main_prompt = loader.build_mechanism_prompt(
        member_id=member_id,
        island_ideology=self.island_ideology,
        error_context=data['error_context'],
        current_mechanisms=str(data['current_mechanisms']),
        modification_attempts=str(data['modification_attempts']),
        report=str(data['report']) if data['report'] else "No analysis available",
        features=data['features'].to_string(),
        relations="\n".join(data['relations']),
        code_memory=data['code_memory'],
        past_performance=data['past_performance'],
        analysis_memory=data['analysis_memory'],
        message_context=data['message_context'],
        base_code=str(self.base_class_code)
    )

    # Add challenge questions
    main_prompt += f"\n\n{base['challenge_questions']}"

    # Add final instruction
    final_prompt = main_prompt + f"\n\n{base['final_instruction_mechanism']}"

    # Call LLM (existing code)
    completion = client.chat.completions.create(
        model=f'{provider}:{model_id}',
        messages=[{"role": "user", "content": final_prompt}]
    )
    # ... rest of existing code
```

## Testing the Integration

1. **Test prompt loading**:
```bash
python -c "from MetaIsland.prompt_loader import get_prompt_loader; print('✓ Prompts loaded successfully')"
```

2. **Test simulation**:
```bash
python -m MetaIsland.metaIsland
```

3. **Verify generated code quality** - Check `generated_code/` directory

## Customizing Prompts

### To add a new action template:

Edit `MetaIsland/prompts/agent_action_templates.yaml`:

```yaml
action_templates:
  - name: "Your New Template"
    description: "What it does"
    code: |
      def agent_action(execution_engine, member_id):
          # Your template code
          pass
```

### To modify base instructions:

Edit `MetaIsland/prompts/base_prompts.yaml`:

```yaml
critical_constraints: |
  [Critical constraints]
  - Your new constraint here
  - Another constraint
```

### To add a new mechanism template:

Edit `MetaIsland/prompts/mechanism_templates.yaml`:

```yaml
contract_templates:
  - name: "New Contract Type"
    description: "What it does"
    code: |
      def propose_modification(execution_engine):
          # Your template code
          pass
```

## Rollback Plan

If issues arise, you can temporarily revert by:

1. Keep the old hardcoded prompt code commented out
2. Add a flag to switch between YAML and hardcoded:

```python
USE_YAML_PROMPTS = True  # Set to False to use old prompts

if USE_YAML_PROMPTS:
    loader = get_prompt_loader()
    prompt = loader.build_action_prompt(...)
else:
    # Old hardcoded prompt code
    prompt = f"""..."""
```

## Performance Considerations

The prompt loader caches YAML files after first load, so there's minimal performance overhead. Initial load adds ~10ms per YAML file.

To pre-warm the cache:
```python
loader = get_prompt_loader()
loader.get_base_prompts()
loader.get_action_templates()
loader.get_mechanism_templates()
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'yaml'`
**Fix**: `pip install pyyaml`

**Issue**: `FileNotFoundError: prompts/base_prompts.yaml`
**Fix**: Ensure YAML files are in `MetaIsland/prompts/` directory

**Issue**: Prompt formatting looks wrong
**Fix**: Check indentation in YAML files - use spaces, not tabs

**Issue**: Variable not being substituted
**Fix**: Ensure you're passing the variable to `build_action_prompt()` or `build_mechanism_prompt()`
