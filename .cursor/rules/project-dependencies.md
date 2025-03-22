# Leviathan Project Dependencies

This document outlines the required dependencies for the Leviathan project.

## Core Dependencies

These dependencies are required for the base Leviathan functionality:

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python     | >= 3.7  | Base language |
| numpy      | >= 1.20.0 | Numerical computations and array operations |
| pandas     | >= 1.3.0 | Data analysis and manipulation |
| matplotlib | >= 3.4.0 | Data visualization |

## Advanced Dependencies

These dependencies are required for the MetaIsland enhanced features:

| Dependency | Version | Purpose |
|------------|---------|---------|
| openai     | >= 0.27.0 | Access to OpenAI API for LLM capabilities |
| aisuite    | latest | AI model access and management |
| asyncio    | stdlib | Asynchronous I/O support |
| dotenv     | >= 0.19.0 | Environment variable management for API keys |

## Development Dependencies

These dependencies are recommended for development:

| Dependency | Version | Purpose |
|------------|---------|---------|
| pytest     | >= 7.0.0 | Testing framework |
| black      | >= 22.0.0 | Code formatting |
| isort      | >= 5.10.0 | Import sorting |
| mypy       | >= 0.950 | Static type checking |

## Installation

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib

# Install advanced dependencies (if needed)
pip install openai python-dotenv aisuite

# Install development dependencies (if needed)
pip install pytest black isort mypy
```

A `requirements.txt` file is available at the root of the project for easy installation:

```bash
pip install -r requirements.txt
```

## Environment Variables

The MetaIsland module requires the following environment variables to be set in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
AISUITE_API_KEY=your_aisuite_api_key
```

## Version Compatibility

The project has been tested with the following environment:
- Python 3.7, 3.8, 3.9, 3.10
- numpy 1.20.3
- pandas 1.3.5
- matplotlib 3.5.1 