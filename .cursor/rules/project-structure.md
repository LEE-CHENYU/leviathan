# Leviathan Project Structure

This document outlines the recommended structure and organization for the Leviathan project.

## Directory Structure

```
leviathan/
├── .cursor/                    # Cursor IDE settings
│   └── rules/                  # Project rules
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore file
├── README.md                   # Project overview
├── requirements.txt            # Project dependencies
├── setup.py                    # Package installation
├── Leviathan/                  # Core module
│   ├── __init__.py
│   ├── README.md               # Module documentation
│   ├── Island.py               # Core simulation class
│   ├── Member.py               # Agent class
│   ├── Land.py                 # Environment class
│   ├── islandExecution.py      # Execution control
│   ├── prompt.py               # LLM prompts
│   ├── settings.py             # Configuration
│   └── ...
├── MetaIsland/                 # Enhanced module
│   ├── __init__.py
│   ├── README.md               # Module documentation
│   ├── metaIsland.py           # Enhanced simulation
│   ├── agent_code_decision.py  # Agent decision logic
│   ├── agent_mechanism_proposal.py # Mechanism proposals
│   ├── analyze.py              # Analysis tools
│   ├── model_router.py         # LLM routing logic
│   └── ...
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── save.py                 # Data persistence
│   └── ...
├── tests/                      # Test directory
│   ├── test_leviathan/         # Tests for core module
│   ├── test_metaisland/        # Tests for enhanced module
│   └── ...
└── examples/                   # Example scripts
    ├── basic_simulation.py
    ├── llm_driven_simulation.py
    └── ...
```

## Module Organization

### Leviathan (Core Module)

The core module contains the basic simulation framework:

- `Island.py`: Main simulation environment class
- `Member.py`: Agent class with properties and behaviors
- `Land.py`: Environment and resource management
- `islandExecution.py`: Execution control for predefined actions
- `settings.py`: Configuration and constants
- `prompt.py`: Templates for LLM interaction

### MetaIsland (Enhanced Module)

The enhanced module expands the core with LLM-driven capabilities:

- `metaIsland.py`: Advanced simulation with LLM integration
- `agent_code_decision.py`: LLM-based agent decision making
- `agent_mechanism_proposal.py`: LLM-based mechanism proposals
- `analyze.py`: Advanced analysis and metrics
- `model_router.py`: LLM provider management

### Utils (Shared Utilities)

Common utilities shared between modules:

- `save.py`: Data persistence functions
- Other helper functions and utilities

## File Organization

### Module Files

Each module file should follow this general organization:

1. Imports
2. Constants
3. Helper functions
4. Main classes
5. Secondary classes

### Class Structure

Classes should be organized with this general structure:

1. Class constants and settings
2. Initialization methods
3. Properties and getters/setters
4. Core functionality methods
5. Helper methods
6. Static/class methods

## Data Management

### Simulation Data

- Simulation data should be saved in a structured format
- Use the specified `save_path` for all outputs
- Organize outputs by simulation run and timestamp

### Output Directories

Generated files should be organized by type:

- `/generated_code`: Code generated during simulation
- `/execution_histories`: Execution history records
- `/analysis`: Analysis outputs
- `/visualizations`: Plots and visual outputs

## Import Conventions

- Absolute imports are preferred for clarity
- Use the following import order:
  1. Standard library imports
  2. Third-party library imports
  3. Project module imports

Example:
```python
import os
import sys
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from Leviathan.Member import Member
from utils.save import path_decorator
``` 