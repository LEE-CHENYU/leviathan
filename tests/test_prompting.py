from MetaIsland.meta_island_prompting import IslandExecutionPromptingMixin


class _PromptingStub(IslandExecutionPromptingMixin):
    pass


def test_summarize_mechanism_code_truncates():
    stub = _PromptingStub()
    body = """def propose_modification(execution_engine):
    \"\"\"Example mechanism\n\nAdds a simple rule.\"\"\"
    x = 1
"""
    code = body + ("    x = x + 1\n" * 5000)
    summary = stub._summarize_mechanism_code(code, max_chars=1000)

    assert len(summary) <= 1000
    assert "Mechanism summary truncated" in summary
