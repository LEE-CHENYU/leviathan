# Leviathan Development Workflow

This document outlines the recommended development workflow for the Leviathan project.

## Development Environment Setup

1. Clone the repository
   ```bash
   git clone https://github.com/username/leviathan.git
   cd leviathan
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. Configure environment variables
   - Create a `.env` file in the project root
   - Add required API keys and configuration values

## Development Workflow

### Feature Development

1. **Branch Creation**
   - Create a new branch for each feature or bug fix
   ```bash
   git checkout -b feature/feature-name
   ```

2. **Development**
   - Follow the [code style guidelines](./code-style.md)
   - Implement tests for new functionality
   - Ensure documentation is updated

3. **Code Quality**
   - Run linters and formatters
   ```bash
   black .
   isort .
   mypy .
   ```

4. **Testing**
   - Run tests for affected modules
   ```bash
   pytest tests/path/to/test_module.py
   ```
   - Run all tests
   ```bash
   pytest
   ```

5. **Commit Changes**
   - Make atomic commits with clear messages
   ```bash
   git add .
   git commit -m "Add feature X"
   ```

6. **Push Changes**
   - Push changes to the remote repository
   ```bash
   git push origin feature/feature-name
   ```

7. **Pull Request**
   - Create a pull request on GitHub
   - Include a clear description of changes
   - Request code review

### Simulation Development

When developing new simulation features:

1. Start with a simple test case
2. Validate results against expected behavior
3. Scale up complexity incrementally
4. Document parameter choices and assumptions
5. Verify that results are reproducible with fixed random seeds

### LLM Integration

When working with LLM features:

1. Test prompts with small examples first
2. Mock API responses during development
3. Keep API keys secure and never commit them
4. Monitor token usage during development
5. Use appropriate error handling for API failures

## Release Process

1. **Version Bump**
   - Update version in `setup.py` and other relevant files
   - Follow [Semantic Versioning](https://semver.org/)

2. **Changelog Update**
   - Document all notable changes in `CHANGELOG.md`
   - Categorize changes (Added, Changed, Fixed, etc.)

3. **Documentation Review**
   - Ensure README and documentation are up-to-date
   - Update API documentation if needed

4. **Final Testing**
   - Run full test suite to verify everything works
   ```bash
   pytest
   ```

5. **Release**
   - Create a release branch
   ```bash
   git checkout -b release/v1.0.0
   ```
   - Make final commits
   - Create a tag
   ```bash
   git tag -a v1.0.0 -m "Version 1.0.0"
   ```
   - Push to GitHub
   ```bash
   git push --follow-tags
   ```

6. **Distribution**
   - Build distribution packages
   ```bash
   python setup.py sdist bdist_wheel
   ```
   - If applicable, upload to PyPI
   ```bash
   twine upload dist/*
   ```

## Continuous Integration

Configure CI/CD to:

1. Run tests on all PRs
2. Apply linters and formatters
3. Build documentation
4. Create releases automatically

## Best Practices

### Simulation Management

- Save simulation states regularly to allow resuming
- Use deterministic random seeds for reproducibility
- Log key metrics and events during simulation
- Create visualization scripts for important metrics

### Performance Optimization

- Profile code to identify bottlenecks
- Use vectorized operations when possible
- Consider multi-processing for parallelizable tasks
- Minimize IO operations during simulation runs

### Collaboration

- Communicate design decisions clearly
- Document complex algorithms
- Use issues for tracking bugs and features
- Request code reviews for all significant changes 