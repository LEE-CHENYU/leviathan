# Leviathan Project Rules

This directory contains the project rules and guidelines for the Leviathan project. These documents establish consistent standards for development, organization, and collaboration.

## Rules Overview

| Rule Document | Description |
|---------------|-------------|
| [Project Dependencies](./project-dependencies.md) | Defines required libraries, versions, and environment setup |
| [Code Style](./code-style.md) | Coding standards, formatting rules, and naming conventions |
| [Project Structure](./project-structure.md) | Directory organization, file placement, and module architecture |
| [Development Workflow](./development-workflow.md) | Development process, testing, and release procedures |

## Using These Rules

These rules serve as guidelines for:

1. **New Developers**: Understand project organization and coding standards
2. **Existing Contributors**: Maintain consistency across the codebase
3. **Reviewers**: Evaluate code contributions against established standards
4. **Project Maintainers**: Guide project evolution and enforce quality

## Project-Specific Guidelines

### Simulation Components

The Leviathan project has specific guidelines for various aspects of the simulation:

- **Agent Design**: Agents should follow the Member class interface
- **Environment Modeling**: Land and resource systems should be modular
- **Decision Making**: Decision functions should be deterministic given the same inputs
- **Relationship Networks**: Relationship data should use standardized formats

### LLM Integration Guidelines

For MetaIsland and LLM-related features:

- Keep prompt templates in separate files for easier management
- Use a consistent approach to LLM API error handling
- Document any prompt engineering techniques used
- Implement caching to reduce API calls during development

## Rule Enforcement

- Use linters and formatters configured according to these rules
- Include rule verification in CI/CD pipelines
- Address rule violations during code reviews
- Update rules when necessary, with team consensus

## Rule Updates

If you need to propose changes to these rules:

1. Create a new branch
2. Edit the relevant rule document
3. Explain the rationale for the change
4. Submit a pull request
5. Discuss with the team
6. Once approved, merge and communicate changes to all contributors 