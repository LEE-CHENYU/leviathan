def calculate_contribution(code_snippet, execution_success):
    # Base contribution for successful code
    base = 1.0 if execution_success else 0.2  
    
    # Additional weight for code complexity
    complexity = len(code_snippet.splitlines()) / 100
    return base + complexity 