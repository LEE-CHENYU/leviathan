import ast
import tempfile

import pytest

from MetaIsland.metaIsland import IslandExecution


@pytest.fixture()
def island():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield IslandExecution(
            init_member_number=1,
            land_shape=(2, 2),
            save_path=tmpdir,
            random_seed=123,
        )


def test_clean_code_string_extracts_fenced_block(island):
    raw = """Here is your code:\n```python\ndef agent_action(execution_engine, member_id):\n    return 1\n```\nThis implementation explains the logic.\n"""
    cleaned = island.clean_code_string(raw)
    assert "This implementation" not in cleaned
    assert cleaned.strip().startswith("def agent_action")
    assert "return 1" in cleaned


def test_clean_code_string_strips_trailing_narrative(island):
    raw = """def agent_action(execution_engine, member_id):\n    return 2\n\nThis implementation provides a summary.\nIt should not be executed.\n"""
    cleaned = island.clean_code_string(raw)
    assert cleaned.strip().endswith("return 2")
    assert "This implementation" not in cleaned
    assert "should not be executed" not in cleaned


def test_clean_code_string_strips_leading_narrative(island):
    raw = (
        "Here is the plan for the modification:\n"
        "1. Define the function\n"
        "2. Return a value\n\n"
        "def agent_action(execution_engine, member_id):\n"
        "    return 3\n"
    )
    cleaned = island.clean_code_string(raw)
    assert cleaned.strip().startswith("def agent_action")
    assert "Here is the plan" not in cleaned
    ast.parse(cleaned)


def test_execute_code_actions_injects_numpy(island):
    island.new_round()
    island.agent_code_by_member = {
        0: """def agent_action(execution_engine, member_id):\n    data = np.array([1, 2, 3])\n    return int(data.sum())\n"""
    }
    island.execute_code_actions()
    errors = island.execution_history["rounds"][-1]["errors"]["agent_code_errors"]
    assert errors == []


def test_clean_code_string_trims_incomplete_function(island):
    raw = (
        "def agent_action(execution_engine, member_id):\n"
        "    return 1\n\n"
        "def incomplete():\n"
        "    \"\"\"Partial docstring\n"
    )
    cleaned = island.clean_code_string(raw)
    assert "def incomplete" not in cleaned
    ast.parse(cleaned)


def test_clean_code_string_drops_unterminated_template(island):
    raw = (
        "def propose_modification(execution_engine):\n"
        "    x = 1\n"
        "    template = '''\n"
        "def foo():\n"
        "    pass\n"
    )
    cleaned = island.clean_code_string(raw)
    assert "def propose_modification" in cleaned
    assert "template = '''" not in cleaned
    ast.parse(cleaned)


def test_clean_code_string_repairs_missing_comma(island):
    raw = (
        "def propose_modification(execution_engine):\n"
        "    config = {\n"
        "        'execution': True\n"
        "        'policy': 'aggressive',\n"
        "    }\n"
        "    return config\n"
    )
    cleaned = island.clean_code_string(raw)
    assert "'execution': True" in cleaned
    assert "'policy': 'aggressive'" in cleaned
    ast.parse(cleaned)


def test_clean_code_string_trims_deep_unterminated_template(island):
    filler = "\n".join("    line = 'x'" for _ in range(80))
    raw = (
        "def propose_modification(execution_engine):\n"
        "    x = 1\n"
        "    template = '''\n"
        f"{filler}\n"
    )
    cleaned = island.clean_code_string(raw)
    assert "template = '''" not in cleaned
    ast.parse(cleaned)
