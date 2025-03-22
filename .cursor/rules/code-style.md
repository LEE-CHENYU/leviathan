# Leviathan Code Style Guidelines

This document outlines the coding style guidelines for the Leviathan project.

## Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for general Python style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length is 88 characters (Black default)
- Use snake_case for variables, functions, and methods
- Use CamelCase for class names
- Use UPPER_CASE for constants

## Naming Conventions

### Classes
- Use CamelCase: `Island`, `Member`, `Land`
- Private methods should start with an underscore: `_attack`, `_bear`

### Methods and Functions
- Use descriptive names that indicate purpose
- Use snake_case: `new_round`, `produce`, `consume`
- Private helper functions should start with an underscore: `_generate_decision_inputs`

### Variables
- Use snake_case: `land_shape`, `init_member_number`
- Class constants should be UPPER_CASE with underscores: `_MAX_VITALITY`, `_REPRODUCE_REQUIREMENT`
- Protected/private attributes should start with an underscore: `_random_seed`, `_create_from_file`

## Documentation

### Docstrings
- Use triple double quotes (`"""`) for docstrings
- Follow the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
- Every module, class, and public method should have a docstring

Example:
```python
def new_round(self):
    """
    Initialize a new round in the simulation.
    
    This method advances the simulation by one step, updating all relevant
    counters and preparing the state for the next round of agent actions.
    
    Returns
    -------
    None
    """
    # method implementation
```

### Comments
- Use comments sparingly and only when necessary
- Focus on explaining "why" rather than "what" or "how"
- Keep comments up-to-date with code changes
- Use Chinese comments for complex domain-specific concepts when necessary

## Type Annotations

- Use type annotations for function parameters and return values
- Use the `typing` module for complex types: `List`, `Dict`, `Optional`, etc.

Example:
```python
from typing import List, Tuple, Optional

def find_neighbors(
    self, 
    member: Member,
    search_range: Optional[int] = None
) -> List[Tuple[Member, float]]:
    # method implementation
```

## Testing

- Write tests for all public methods
- Test files should be in a `tests` directory, mirroring the structure of the code
- Test file names should match the module they test, with a `test_` prefix
- Use pytest for testing

## Linting and Formatting

- Use Black for code formatting
- Use isort for sorting imports
- Use mypy for type checking

## Git Commit Messages

- Use imperative mood ("Add feature" not "Added feature")
- Start with a capital letter
- Keep the subject line under 50 characters
- Wrap the body at 72 characters
- Separate subject from body with a blank line
- Use the body to explain what and why, not how 