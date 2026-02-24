import sys
import traceback

def execute_code_safely(code):
    print("\n=== Executing Code Safely ===")
    print(f"Code to execute: {code[:200]}...")
    restricted_globals = {
        '__builtins__': {
            'len': len,
            'range': range,
            'print': print,  # Make print available if needed
            'int': int,
            'str': str,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            # Add more safe builtins for basic operations
        }
    }
    try:
        exec(code, restricted_globals)
        print("Code execution successful")
        return True
    except Exception as e:
        print(f"=== Code Execution Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        return False 