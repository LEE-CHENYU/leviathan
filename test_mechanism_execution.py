import json
from pathlib import Path
import textwrap

import pytest

from MetaIsland.metaIsland import IslandExecution


def _dedent(code: str) -> str:
    return textwrap.dedent(code).strip()


def test_execute_mechanism_modifications_respects_approved(tmp_path):
    exec_engine = IslandExecution(2, (3, 3), str(tmp_path), random_seed=1)
    exec_engine.new_round()

    approved_mod = {
        'member_id': 0,
        'code': _dedent(
            """
            def propose_modification(self):
                self.approved_flag = True
            """
        ),
    }
    rejected_mod = {
        'member_id': 1,
        'code': _dedent(
            """
            def propose_modification(self):
                self.rejected_flag = True
            """
        ),
    }

    round_record = exec_engine.execution_history['rounds'][-1]
    round_record['mechanism_modifications']['attempts'] = [approved_mod, rejected_mod]

    exec_engine.execute_mechanism_modifications(approved=[approved_mod])

    mods_record = round_record['mechanism_modifications']
    assert mods_record.get('approved_count') == 1
    assert mods_record.get('approved_ids') == [0]

    assert getattr(exec_engine, 'approved_flag', False) is True
    assert getattr(exec_engine, 'rejected_flag', False) is False

    executed = round_record['mechanism_modifications']['executed']
    assert len(executed) == 1
    assert executed[0].get('member_id') == 0


def test_mechanism_execution_log_serializable(tmp_path):
    exec_engine = IslandExecution(1, (2, 2), str(tmp_path), random_seed=2)
    exec_engine.new_round()

    mod = {
        'member_id': 0,
        'code': _dedent(
            """
            def propose_modification(self):
                self.log_flag = True
            """
        ),
    }

    exec_engine.execution_history['rounds'][-1]['mechanism_modifications']['attempts'] = [mod]

    exec_engine.execute_mechanism_modifications(approved=[mod])

    assert getattr(exec_engine, 'log_flag', False) is True

    mech_dir = Path(exec_engine.mechanism_code_path)
    json_files = list(mech_dir.glob('mechanism_execution_*_round_*_*.json'))
    assert json_files, "Expected mechanism execution JSON log"

    with json_files[-1].open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    assert payload.get('member_id') == 0
    assert 'class_attributes' in payload


def test_round_metrics_include_mechanism_counts(tmp_path):
    exec_engine = IslandExecution(1, (2, 2), str(tmp_path), random_seed=3)
    exec_engine.new_round()

    mod = {
        'member_id': 0,
        'code': _dedent(
            """
            def propose_modification(self):
                self.metric_flag = True
            """
        ),
    }

    round_record = exec_engine.execution_history['rounds'][-1]
    round_record['mechanism_modifications']['attempts'] = [mod]

    exec_engine.execute_mechanism_modifications(approved=[mod])
    exec_engine.execute_code_actions()

    metrics = round_record.get('round_metrics') or {}
    assert metrics.get('mechanism_attempted_count') == 1
    assert metrics.get('mechanism_approved_count') == 1
    assert metrics.get('mechanism_executed_count') == 1
    assert metrics.get('mechanism_error_count') == 0


def test_round_metrics_include_plan_feasibility(tmp_path):
    exec_engine = IslandExecution(2, (3, 3), str(tmp_path), random_seed=4)
    exec_engine.new_round()

    member = exec_engine.current_members[0]
    member.land_num = 0
    member.owned_land = []

    analysis_text = textwrap.dedent(
        """
        Strategy update.
        ```json
        {
          "hypothesis": "test plan feasibility",
          "baseline_signature": ["bear", "message"],
          "variation_signature": ["attack"],
          "success_metrics": ["delta_survival"],
          "guardrails": ["avoid death"],
          "coordination": ["message ally"],
          "memory_note": "feasibility test",
          "diversity_note": "avoid monoculture",
          "confidence": 0.5
        }
        ```
        """
    ).strip()
    exec_engine._record_analysis_card(0, analysis_text)

    code = _dedent(
        """
        def agent_action(execution_engine, member_id):
            execution_engine.send_message(member_id, 1, "hi")
        """
    )
    exec_engine.agent_code_by_member = {0: code}
    exec_engine.execute_code_actions()

    metrics = exec_engine.execution_history['rounds'][-1].get('round_metrics') or {}
    assert metrics.get('plan_tag_total') == 2
    assert metrics.get('plan_ineligible_tag_count') == 1
    assert metrics.get('plan_feasible_tag_total') == 1
    assert metrics.get('plan_only_tag_count') == 0
    assert metrics.get('plan_feasibility_samples') == 1
    assert metrics.get('plan_ineligible_tag_rate') == pytest.approx(0.5)
    assert metrics.get('plan_only_tag_rate') == pytest.approx(0.0)


def test_round_metrics_include_agent_error_tags(tmp_path):
    exec_engine = IslandExecution(1, (2, 2), str(tmp_path), random_seed=5)
    exec_engine.new_round()

    code = _dedent(
        """
        def agent_action(execution_engine, member_id):
            if False:
                execution_engine.attack(member_id, member_id)
            raise RuntimeError("boom")
        """
    )
    exec_engine.agent_code_by_member = {0: code}
    exec_engine.execute_code_actions()

    metrics = exec_engine.execution_history['rounds'][-1].get('round_metrics') or {}
    assert metrics.get('agent_code_error_count') == 1
    assert metrics.get('agent_code_error_rate') == pytest.approx(1.0)
    tag_counts = metrics.get('agent_code_error_tag_counts') or {}
    assert tag_counts.get('attack') == 1
