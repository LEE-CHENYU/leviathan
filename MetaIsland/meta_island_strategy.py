from typing import List, Tuple, Optional
import ast
import re
import numpy as np
import pandas as pd
import json
from collections import Counter

class StrategyMemory(dict):
    """Dict-backed strategy memory that supports list-style append."""

    def append(self, item) -> None:
        notes = self.get("notes")
        if not isinstance(notes, list):
            notes = []
        notes.append(item)
        self["notes"] = notes




class IslandExecutionStrategyMixin:
    def parse_relationship_matrix(self, relationship_dict):
        """
        Parse and return a human-readable summary of the relationship matrices.
        
        :param relationship_dict: A dictionary with keys like 'victim', 'benefit', 'benefit_land'
                                 each containing a NxN numpy array of relationships.
        :return: A list of strings describing the relationships.
        """
        summary = []
        rel_map = {
            'victim':      "member_{i} was attacked by member_{j}",
            'benefit':     "member_{i} gave a benefit to member_{j}",
            'benefit_land':"member_{i} gave land to member_{j}"
        }
        
        for relation_type, matrix in relationship_dict.items():
            if relation_type not in rel_map:
                continue
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    # Filter out invalid or zero entries
                    if not np.isnan(val) and val != 0:
                        # Construct a description
                        member_i_id = self.current_members[i].id
                        member_j_id = self.current_members[j].id
                        statement = (f"{rel_map[relation_type]} "
                                     f"(value={val:.2f})")
                        # Replace {i} with actual index+1 (or keep zero-based)
                        # Same for {j}
                        statement = statement.format(i=member_i_id, j=member_j_id)
                        summary.append(statement)
        
        return summary
    
    def get_current_member_features(self) -> pd.DataFrame:
        """Collect features for all current members"""
        feature_rows = []
        
        for member in self.current_members:
            # Get self attributes
            feature_row = {
                "self_productivity": member.overall_productivity,
                "self_vitality": member.vitality, 
                "self_cargo": member.cargo,
                "self_age": member.age,
                "self_neighbor": len(member.current_clear_list),
                "member_id": member.id
            }
            feature_rows.append(feature_row)
                
        return pd.DataFrame(feature_rows)

    def clean_code_string(self, code_str: str) -> str:
        """Remove markdown code block markers and clean up the code string."""
        if code_str is None:
            return ""
        code_str = str(code_str)

        # Prefer the first fenced code block if present.
        if "```" in code_str:
            fenced_blocks = re.findall(r"```(?:[\w+-]+)?\s*\n(.*?)```", code_str, flags=re.DOTALL)
            if fenced_blocks:
                code_str = fenced_blocks[0]
            else:
                code_str = code_str.replace('```python', '').replace('```', '')

        # Remove any leading or trailing whitespace
        lines = code_str.split('\n')
        lines = [line.rstrip() for line in lines]

        # Remove any empty lines at start/end while preserving internal empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Strip leading/trailing narrative lines that are not valid top-level Python.
        def _looks_like_toplevel_code(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            if stripped[0] in ("\"", "'"):
                return True
            if stripped.startswith("#"):
                return True
            for prefix in (
                "def ",
                "class ",
                "import ",
                "from ",
                "if ",
                "for ",
                "while ",
                "with ",
                "async ",
                "@",
                "try:",
                "elif ",
                "else:",
                "except",
                "finally:",
                "pass",
            ):
                if stripped.startswith(prefix):
                    return True
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
                return True
            return False

        while lines:
            line = lines[0]
            if not line.strip():
                lines.pop(0)
                continue
            if _looks_like_toplevel_code(line):
                break
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)

        idx = len(lines) - 1
        while idx >= 0:
            line = lines[idx]
            if not line.strip():
                idx -= 1
                continue
            if line[: len(line) - len(line.lstrip())]:
                break
            if _looks_like_toplevel_code(line):
                break
            idx -= 1

        if idx < len(lines) - 1:
            lines = lines[: idx + 1]
            while lines and not lines[-1].strip():
                lines.pop()

        cleaned = '\n'.join(lines)
        if not cleaned:
            return ""

        def _attempt_parse(candidate: str):
            try:
                ast.parse(candidate)
                return None
            except SyntaxError as err:
                return err

        def _syntax_trim_candidate(err: SyntaxError, line_count: int) -> bool:
            msg = getattr(err, "msg", "") or ""
            lineno = getattr(err, "lineno", None)
            tail_window = max(3, int(line_count * 0.1))
            near_end = lineno is None or lineno >= max(1, line_count - tail_window)
            eof_hint = (
                "EOF while scanning" in msg
                or "unexpected EOF" in msg
                or "EOL while scanning" in msg
            )
            indent_hint = "expected an indented block" in msg
            unterminated_hint = (
                "unterminated string literal" in msg
                or "triple-quoted string literal" in msg
            )
            return near_end or eof_hint or indent_hint or unterminated_hint

        def _maybe_insert_missing_comma(source_lines, err):
            msg = getattr(err, "msg", "") or ""
            if "comma" not in msg.lower():
                return None
            lineno = getattr(err, "lineno", None)
            if lineno is None:
                return None

            def _eligible(line: str) -> bool:
                stripped = line.strip()
                if not stripped or stripped.endswith(","):
                    return False
                if stripped.startswith(
                    (
                        "#",
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "with ",
                        "try",
                        "except",
                        "finally",
                        "return",
                        "raise",
                        "yield",
                        "break",
                        "continue",
                        "import ",
                        "from ",
                        "pass",
                    )
                ):
                    return False
                if ":" in stripped:
                    return True
                if stripped.endswith((")", "]", "}", "\"", "'")):
                    return True
                return False

            candidate_lines = list(source_lines)
            for target in (lineno - 1, lineno - 2):
                if target < 0 or target >= len(candidate_lines):
                    continue
                line = candidate_lines[target]
                if not _eligible(line):
                    continue
                candidate_lines[target] = line + ","
                candidate = "\n".join(candidate_lines)
                if _attempt_parse(candidate) is None:
                    return candidate
                candidate_lines[target] = line
            return None

        def _trim_to_parseable(source_lines):
            trimmed_lines = list(source_lines)
            max_trim = len(trimmed_lines)
            for _ in range(max_trim):
                if not trimmed_lines:
                    return ""
                trimmed_lines.pop()
                while trimmed_lines and not trimmed_lines[-1].strip():
                    trimmed_lines.pop()
                if not trimmed_lines:
                    return ""
                candidate = "\n".join(trimmed_lines)
                if _attempt_parse(candidate) is None:
                    return candidate
            return ""

        parse_err = _attempt_parse(cleaned)
        if parse_err is None:
            return cleaned

        repaired = _maybe_insert_missing_comma(lines, parse_err)
        if repaired is not None:
            return repaired

        if not _syntax_trim_candidate(parse_err, len(lines)):
            return cleaned

        trimmed = _trim_to_parseable(lines)
        return trimmed or cleaned

    def _truncate_message(self, message: str, limit: int = 120) -> str:
        """Truncate messages for concise logging/memory."""
        if message is None:
            return ""
        text = str(message)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _resolve_member_index(self, member_ref) -> Optional[int]:
        """Resolve a member reference (index, id, or Member) to current_members index."""
        if member_ref is None:
            return None
        if hasattr(member_ref, "id"):
            member_ref = member_ref.id
        if isinstance(member_ref, int) and 0 <= member_ref < len(self.current_members):
            return member_ref
        for idx, member in enumerate(self.current_members):
            if member.id == member_ref:
                return idx
        return None

    def _resolve_member_stable_id(self, member_ref) -> Optional[int]:
        """Resolve a member reference (index, id, or Member) to stable member.id."""
        if member_ref is None:
            return None
        if hasattr(member_ref, "id"):
            return member_ref.id
        if isinstance(member_ref, int) and 0 <= member_ref < len(self.current_members):
            return self.current_members[member_ref].id
        for member in self.current_members:
            if member.id == member_ref:
                return member_ref
        return None

    def _get_member_history(self, store: dict, member_ref) -> Tuple[Optional[int], list]:
        """Return (stable_id, history_list) with light migration from index keys."""
        stable_id = self._resolve_member_stable_id(member_ref)
        if stable_id is None:
            return None, []
        if stable_id in store:
            return stable_id, store[stable_id]
        if isinstance(member_ref, int) and member_ref in store and member_ref != stable_id:
            store[stable_id] = store[member_ref]
            return stable_id, store[stable_id]
        return stable_id, []

    def _get_round_member_entry(self, round_data: dict, key: str, member_ref):
        """Fetch per-member round_data entry using stable id with index fallback."""
        if not round_data or not key:
            return None
        bucket = round_data.get(key, {})
        if not isinstance(bucket, dict):
            return None
        stable_id = self._resolve_member_stable_id(member_ref)
        if stable_id is None:
            return None
        if stable_id in bucket:
            return bucket[stable_id]
        if isinstance(member_ref, int) and member_ref in bucket:
            return bucket[member_ref]
        return None

    def _ensure_strategy_memory_appendable(self, member) -> None:
        """Ensure strategy_memory supports .append for agent-generated code."""
        if member is None:
            return
        if not hasattr(member, "strategy_memory") or member.strategy_memory is None:
            member.strategy_memory = StrategyMemory()
            return
        mem = member.strategy_memory
        if isinstance(mem, StrategyMemory):
            return
        if isinstance(mem, dict):
            member.strategy_memory = StrategyMemory(mem)

    def _auto_update_strategy_memory(
        self,
        member,
        round_num: int,
        signature: tuple,
        metrics: dict,
        context_tags: Optional[dict] = None,
        max_items: int = 12,
    ) -> None:
        """Append a compact auto note to strategy_memory for learning continuity."""
        if member is None:
            return

        def _fmt(value, decimals: int) -> str:
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return f"{0.0:.{decimals}f}"

        sig_text = self._format_signature(signature) if signature else "none"
        ctx_text = self._format_context_tags(context_tags) if context_tags else "none"
        note = (
            f"auto r{round_num} sig={sig_text} "
            f"d_surv={_fmt(metrics.get('delta_survival'), 2)} "
            f"d_vit={_fmt(metrics.get('delta_vitality'), 1)} "
            f"d_cargo={_fmt(metrics.get('delta_cargo'), 1)} "
            f"d_rel={_fmt(metrics.get('delta_relation_balance'), 2)} "
            f"d_land={_fmt(metrics.get('delta_land'), 1)} "
            f"ctx={ctx_text}"
        )

        if not hasattr(member, "strategy_memory") or member.strategy_memory is None:
            member.strategy_memory = {"auto_notes": []}

        mem = member.strategy_memory
        if isinstance(mem, dict):
            bucket = mem.get("auto_notes")
            if not isinstance(bucket, list):
                bucket = []
                mem["auto_notes"] = bucket
            bucket.append(note)
            if len(bucket) > max_items:
                del bucket[:-max_items]
        elif isinstance(mem, list):
            mem.append(note)
            if len(mem) > max_items:
                del mem[:-max_items]
        else:
            member.strategy_memory = {"auto_notes": [str(mem), note]}

    def _coerce_card_text(self, value) -> str:
        """Coerce a strategy card field into a compact string."""
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).strip()
        return text

    def _coerce_card_list(self, value) -> List[str]:
        """Coerce a strategy card list field into a list of strings."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, str):
            if "," in value:
                items = value.split(",")
            elif "|" in value:
                items = value.split("|")
            else:
                text = value.strip()
                return [text] if text else []
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def _coerce_card_signature(self, value) -> List[str]:
        """Coerce a signature field into tag strings, splitting '+' delimited combos."""
        items = self._coerce_card_list(value)
        expanded: List[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            if "+" in text:
                parts = [part.strip() for part in text.split("+") if part.strip()]
                expanded.extend(parts)
            else:
                expanded.append(text)
        return expanded

    def _normalize_strategy_card(self, raw: dict) -> Optional[dict]:
        """Normalize strategy card keys to a consistent schema."""
        if not isinstance(raw, dict):
            return None

        def pick(*keys):
            for key in keys:
                if key in raw:
                    return raw[key]
            return None

        card = {
            "hypothesis": self._coerce_card_text(
                pick("hypothesis", "hyp", "thesis")
            ),
            "baseline_signature": self._coerce_card_signature(
                pick("baseline_signature", "baseline_tags", "baseline", "baseline_action")
            ),
            "variation_signature": self._coerce_card_signature(
                pick("variation_signature", "variation_tags", "variation", "variant_action")
            ),
            "success_metrics": self._coerce_card_list(
                pick("success_metrics", "metrics", "success", "targets")
            ),
            "guardrails": self._coerce_card_list(
                pick("guardrails", "constraints", "stops", "stop_conditions")
            ),
            "coordination": self._coerce_card_list(
                pick("coordination", "coordination_asks", "messages", "coordination_plan")
            ),
            "memory_note": self._coerce_card_text(
                pick("memory_note", "note", "memory", "log_note")
            ),
            "diversity_note": self._coerce_card_text(
                pick("diversity_note", "diversity", "anti_monoculture")
            ),
        }

        confidence = pick("confidence", "conf", "certainty")
        if confidence is not None:
            try:
                card["confidence"] = float(confidence)
            except (TypeError, ValueError):
                card["confidence"] = self._coerce_card_text(confidence)

        if not any(value for value in card.values()):
            return None
        return card

    def _extract_strategy_card(self, text: str) -> Optional[dict]:
        """Extract a JSON strategy card from analysis text."""
        if not text:
            return None
        match = re.search(
            r"```json\s*(\{.*?\})\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            match = re.search(
                r"STRATEGY_CARD_JSON\s*:?\s*(\{.*?\})",
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
        if not match:
            return None
        raw = match.group(1).strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return None
        if not isinstance(parsed, dict):
            return None
        return self._normalize_strategy_card(parsed)

    def _format_strategy_card_brief(self, card: dict, limit: int = 180) -> str:
        """Format a compact, single-line summary of a strategy card."""
        if not card:
            return ""
        parts = []
        hypothesis = card.get("hypothesis")
        if hypothesis:
            parts.append(f"hyp={hypothesis}")
        baseline = card.get("baseline_signature") or []
        if baseline:
            parts.append(f"base={','.join(baseline)}")
        variation = card.get("variation_signature") or []
        if variation:
            parts.append(f"var={','.join(variation)}")
        success = card.get("success_metrics") or []
        if success:
            parts.append(f"success={','.join(success)}")
        guardrails = card.get("guardrails") or []
        if guardrails:
            parts.append(f"guards={','.join(guardrails)}")
        diversity_note = card.get("diversity_note")
        if diversity_note:
            parts.append(f"diversity={diversity_note}")
        confidence = card.get("confidence")
        if confidence is not None:
            parts.append(f"conf={confidence}")
        summary = " | ".join(parts)
        return self._truncate_message(summary, limit=limit)

    def _append_strategy_report(
        self,
        member,
        note: str,
        key: str = "reports",
        max_items: int = 6,
    ) -> None:
        """Append a short analysis note to member.strategy_memory."""
        if member is None or not note:
            return
        if not hasattr(member, "strategy_memory") or member.strategy_memory is None:
            member.strategy_memory = {key: []}
        mem = member.strategy_memory
        if isinstance(mem, dict):
            bucket = mem.get(key)
            if not isinstance(bucket, list):
                bucket = []
            bucket.append(note)
            if len(bucket) > max_items:
                del bucket[:-max_items]
            mem[key] = bucket
        elif isinstance(mem, list):
            mem.append(note)
            if len(mem) > max_items:
                del mem[:-max_items]
        else:
            member.strategy_memory = {key: [str(mem), note]}

    def _record_analysis_card(self, member_id: int, analysis_text: str) -> None:
        """Parse and store a strategy card from analysis output."""
        card = self._extract_strategy_card(analysis_text)
        if not card:
            return
        if not self.execution_history.get("rounds"):
            return
        round_data = self.execution_history["rounds"][-1]
        stable_id = self._resolve_member_stable_id(member_id)
        if stable_id is None:
            return
        round_data.setdefault("analysis_cards", {})[stable_id] = card
        member = None
        member_index = self._resolve_member_index(member_id)
        if member_index is not None:
            try:
                member = self.current_members[member_index]
            except Exception:
                member = None
        round_num = round_data.get("round_number", len(self.execution_history["rounds"]))
        memory_note = card.get("memory_note")
        if memory_note:
            note = f"analysis r{round_num} | {memory_note}"
        else:
            brief = self._format_strategy_card_brief(card, limit=140)
            note = f"analysis r{round_num} | {brief}" if brief else f"analysis r{round_num}"
        note = self._truncate_message(note, limit=180)
        self._append_strategy_report(member, note)

    def get_analysis_card_summary(self, member_id: int, window: int = 3) -> str:
        """Summarize recent analysis strategy cards for prompt guidance."""
        if not self.execution_history.get("rounds"):
            return "No analysis cards available."
        window = max(1, int(window))
        cards = []
        for round_data in self.execution_history["rounds"][-window:]:
            card = self._get_round_member_entry(round_data, "analysis_cards", member_id)
            if card:
                round_num = round_data.get("round_number")
                cards.append((round_num, card))
        if not cards:
            return "No analysis cards available."
        lines = ["Recent analysis strategy cards:"]
        for round_num, card in cards:
            label = f"Round {round_num}" if round_num is not None else "Round ?"
            summary = self._format_strategy_card_brief(card, limit=160)
            lines.append(f"- {label}: {summary if summary else 'card parsed'}")
        return "\n".join(lines)

    def get_experiment_summary(
        self,
        member_id: int,
        window: int = 6,
        min_samples: int = 2,
    ) -> str:
        """Summarize baseline/variation experiment outcomes from recent actions."""
        _, memory = self._get_member_history(self.code_memory, member_id)
        if not memory:
            return "No experiment outcomes yet."

        experiments = [mem for mem in memory if mem.get("experiment")]
        if not experiments:
            return "No experiment outcomes yet."

        window = max(1, int(window))
        recent = experiments[-window:]
        if not recent:
            return "No experiment outcomes yet."

        labels = ("baseline", "variation", "unmatched")
        perf_by_label = {label: [] for label in labels}
        match_scores = {label: [] for label in labels}

        for mem in recent:
            exp = mem.get("experiment") or {}
            label = exp.get("label", "unmatched")
            if label not in perf_by_label:
                label = "unmatched"
            perf_by_label[label].append(self._get_memory_performance(mem))
            score = exp.get("match_score")
            if score is not None:
                try:
                    match_scores[label].append(float(score))
                except (TypeError, ValueError):
                    pass

        def _avg(values: list) -> Optional[float]:
            if not values:
                return None
            return sum(values) / len(values)

        base_avg = _avg(perf_by_label["baseline"])
        var_avg = _avg(perf_by_label["variation"])
        other_avg = _avg(perf_by_label["unmatched"])

        lines = [
            "Experiment outcomes (recent baseline/variation tests):",
            f"- Window: {len(recent)} actions; "
            f"baseline {len(perf_by_label['baseline'])}, "
            f"variation {len(perf_by_label['variation'])}, "
            f"other {len(perf_by_label['unmatched'])}",
        ]

        if base_avg is not None:
            lines.append(
                f"- Baseline avg delta_survival: {base_avg:.2f} "
                f"(n={len(perf_by_label['baseline'])})"
            )
        if var_avg is not None:
            lines.append(
                f"- Variation avg delta_survival: {var_avg:.2f} "
                f"(n={len(perf_by_label['variation'])})"
            )
        if other_avg is not None and perf_by_label["unmatched"]:
            lines.append(
                f"- Unmatched avg delta_survival: {other_avg:.2f} "
                f"(n={len(perf_by_label['unmatched'])})"
            )

        total_count = len(recent)
        matched_count = (
            len(perf_by_label["baseline"]) + len(perf_by_label["variation"])
        )
        match_rate = matched_count / total_count if total_count else 0.0
        avg_match_score = None
        match_pool = match_scores.get("baseline", []) + match_scores.get("variation", [])
        if match_pool:
            avg_match_score = sum(match_pool) / len(match_pool)
        if total_count:
            lines.append(
                f"- Plan match rate: {match_rate:.2f} "
                f"(baseline {len(perf_by_label['baseline'])}, "
                f"variation {len(perf_by_label['variation'])}, "
                f"unmatched {len(perf_by_label['unmatched'])})"
            )
        if avg_match_score is not None:
            lines.append(
                f"- Avg match score (matched): {avg_match_score:.2f}"
            )

        best_label = None
        best_avg = None
        for label, avg in (("baseline", base_avg), ("variation", var_avg)):
            if avg is None:
                continue
            if best_avg is None or avg > best_avg:
                best_avg = avg
                best_label = label
        if best_label is not None:
            lines.append(
                f"- Best recent plan: {best_label} "
                f"(avg delta_survival {best_avg:.2f})"
            )

        low_sample = (
            len(perf_by_label["baseline"]) < min_samples
            or len(perf_by_label["variation"]) < min_samples
        )
        if low_sample:
            lines.append(
                "- Low sample in recent tests; keep bounded variation to avoid premature convergence."
            )
        if total_count and match_rate < 0.5:
            lines.append(
                "- Plan adherence low; follow baseline/variation tags from analysis to keep experiments interpretable."
            )

        return "\n".join(lines)

    def _collect_strategy_notes(
        self,
        member,
        max_items: int = 3,
        limit: int = 160,
    ) -> List[str]:
        """Collect compact strategy_memory notes from the member for logging."""
        if member is None or not hasattr(member, "strategy_memory"):
            return []

        raw = getattr(member, "strategy_memory")
        notes: List[str] = []

        def _stringify(value) -> str:
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=True, default=str)
            except (TypeError, ValueError):
                return str(value)

        def _add(prefix: Optional[str], value) -> None:
            text = self._truncate_message(_stringify(value), limit=limit)
            if prefix:
                notes.append(f"{prefix}: {text}")
            else:
                notes.append(text)

        if isinstance(raw, dict):
            preferred_keys = (
                "notes",
                "tactics",
                "experiments",
                "hypotheses",
                "outcomes",
                "reports",
                "auto_notes",
            )
            for key in preferred_keys:
                if key not in raw:
                    continue
                value = raw.get(key)
                if isinstance(value, list) and value:
                    _add(key, value[-1])
                elif isinstance(value, dict) and value:
                    last_key = next(reversed(value))
                    _add(f"{key}.{last_key}", value[last_key])
                elif value is not None:
                    _add(key, value)
                if len(notes) >= max_items:
                    break
            if not notes:
                _add("strategy_memory", raw)
        elif isinstance(raw, list):
            for item in raw[-max_items:]:
                _add(None, item)
        elif raw is not None:
            _add(None, raw)

        return notes[:max_items]

    def _parse_message_sender(self, message: str) -> Optional[int]:
        """Parse sender id from message format 'From member_X: ...'."""
        if not message:
            return None
        prefix = "From member_"
        if not str(message).startswith(prefix):
            return None
        remainder = str(message)[len(prefix):]
        sender_token = remainder.split(":", 1)[0].strip()
        try:
            return int(sender_token)
        except ValueError:
            return None

    def _summarize_communication(
        self,
        member_id: int,
        window_rounds: int = 3,
        sample_limit: int = 2,
    ) -> str:
        """Summarize recent message exchanges for coordination context."""
        if not self.execution_history.get('rounds'):
            return "No communication history available."

        total_rounds = len(self.execution_history['rounds'])
        window_rounds = max(1, int(window_rounds))
        start_round = max(1, total_rounds - window_rounds + 1)

        sent_total = 0
        received_total = 0
        recipients = set()
        senders = set()
        recent_received = []
        recent_sent = []

        def push_recent(bucket, item):
            if sample_limit <= 0:
                return
            bucket.append(item)
            if len(bucket) > sample_limit:
                bucket.pop(0)

        for round_data in self.execution_history['rounds'][start_round - 1:]:
            comm = self._get_round_member_entry(round_data, "agent_messages", member_id)
            if not comm:
                continue
            for msg in comm.get('received', []) or []:
                received_total += 1
                sender_id = self._parse_message_sender(msg)
                if sender_id is not None:
                    senders.add(sender_id)
                push_recent(recent_received, self._truncate_message(msg))
            for recipient_id, msg in comm.get('sent', []) or []:
                sent_total += 1
                recipients.add(recipient_id)
                msg_text = self._truncate_message(msg)
                if recipient_id is not None:
                    msg_text = f"member_{recipient_id}: {msg_text}"
                push_recent(recent_sent, msg_text)

        if sent_total == 0 and received_total == 0:
            return (
                f"Communication summary (rounds {start_round}-{total_rounds}): "
                "no messages exchanged."
            )

        lines = [
            "Communication summary (recent rounds):",
            f"- Window rounds: {start_round}-{total_rounds}",
            f"- Sent: {sent_total} messages to {len(recipients)} unique recipients",
            f"- Received: {received_total} messages from {len(senders)} unique senders",
        ]

        if recipients:
            recipient_list = sorted(recipients)
            display = ", ".join(f"member_{rid}" for rid in recipient_list[:5])
            if len(recipient_list) > 5:
                display += ", ..."
            lines.append(f"- Recent recipients: {display}")
        if senders:
            sender_list = sorted(senders)
            display = ", ".join(f"member_{sid}" for sid in sender_list[:5])
            if len(sender_list) > 5:
                display += ", ..."
            lines.append(f"- Recent senders: {display}")
        if recent_received:
            lines.append(f"- Recent received: {recent_received}")
        if recent_sent:
            lines.append(f"- Recent sent: {recent_sent}")

        return "\n".join(lines)

    def _collect_contract_partner_stats(self) -> dict:
        """Aggregate partner mix and status counts for contracts by party."""
        stats = {
            "partner_counts": {},
            "status_counts": {},
            "partner_unique": {},
            "partner_top_share": {},
            "partner_top_partner": {},
            "avg_unique_partners": 0.0,
            "avg_top_partner_share": 0.0,
        }
        if not hasattr(self, "contracts"):
            return stats

        partner_counts = {}
        status_counts = {}
        for contract in self.contracts.contracts.values():
            parties = contract.get("parties", []) or []
            status = contract.get("status", "unknown")
            for party in parties:
                status_counts.setdefault(party, Counter())[status] += 1
                for other in parties:
                    if other == party:
                        continue
                    partner_counts.setdefault(party, Counter())[other] += 1

        partner_unique = {}
        partner_top_share = {}
        partner_top_partner = {}
        for party, counts in partner_counts.items():
            partner_unique[party] = len(counts)
            if counts:
                top_partner, top_count = counts.most_common(1)[0]
                total_links = sum(counts.values())
                partner_top_partner[party] = top_partner
                partner_top_share[party] = (
                    float(top_count) / total_links if total_links else 0.0
                )
            else:
                partner_top_partner[party] = None
                partner_top_share[party] = 0.0

        member_ids = [m.id for m in getattr(self, "current_members", []) or []]
        if member_ids:
            unique_vals = [partner_unique.get(mid, 0) for mid in member_ids]
            share_vals = [partner_top_share.get(mid, 0.0) for mid in member_ids]
            stats["avg_unique_partners"] = float(np.mean(unique_vals)) if unique_vals else 0.0
            stats["avg_top_partner_share"] = float(np.mean(share_vals)) if share_vals else 0.0

        stats["partner_counts"] = partner_counts
        stats["status_counts"] = status_counts
        stats["partner_unique"] = partner_unique
        stats["partner_top_share"] = partner_top_share
        stats["partner_top_partner"] = partner_top_partner
        return stats

    def _summarize_contract_activity(self, member_id: int, max_partners: int = 3) -> str:
        """Summarize contract involvement and partner diversity for a member."""
        if not hasattr(self, "contracts"):
            return "Contract activity: unavailable."

        stable_id = self._resolve_member_stable_id(member_id)
        party_id = stable_id if stable_id is not None else member_id

        contracts = [
            contract for contract in self.contracts.contracts.values()
            if party_id in (contract.get("parties") or [])
        ]
        if not contracts:
            return "Contract activity: none."

        status_counts = Counter()
        for contract in contracts:
            status_counts[contract.get("status", "unknown")] += 1

        partner_stats = self._collect_contract_partner_stats()
        partner_counts = partner_stats.get("partner_counts", {}).get(party_id, Counter())
        partner_unique = partner_stats.get("partner_unique", {}).get(party_id, 0)
        top_partner = partner_stats.get("partner_top_partner", {}).get(party_id)
        top_share = partner_stats.get("partner_top_share", {}).get(party_id, 0.0)

        partner_text = "none"
        if partner_counts:
            partner_text = ", ".join(
                f"member_{pid} (n={count})"
                for pid, count in partner_counts.most_common(max(1, int(max_partners)))
            )

        lines = [
            "Contract activity (your involvement):",
            f"- contracts: total {len(contracts)} | pending {status_counts.get('pending', 0)} "
            f"| active {status_counts.get('active', 0)} | completed {status_counts.get('completed', 0)} "
            f"| failed {status_counts.get('failed', 0)}",
            f"- partners: {partner_unique} unique; top partner "
            f"{'member_' + str(top_partner) if top_partner is not None else 'none'} "
            f"(share {top_share:.2f})",
            f"- partner mix: {partner_text}",
        ]

        if top_share >= 0.6 and partner_unique > 1:
            lines.append(
                "- concentration risk: consider diversifying partners or terms."
            )

        return "\n".join(lines)

    def _peek_messages(
        self,
        member_id: int,
        round_num: Optional[int] = None,
        create_snapshot: bool = True,
    ) -> list:
        """Return unread messages for the member without clearing them."""
        if round_num is None:
            round_num = len(self.execution_history.get("rounds", []))
        member_key = self._resolve_member_stable_id(member_id)
        if member_key is None:
            return []
        if (
            isinstance(member_id, int)
            and member_id != member_key
            and member_id in self.messages
            and member_key not in self.messages
        ):
            self.messages[member_key] = self.messages.pop(member_id)
        if (
            isinstance(member_id, int)
            and member_id != member_key
            and member_id in self._message_snapshot_round
            and member_key not in self._message_snapshot_round
        ):
            self._message_snapshot_round[member_key] = self._message_snapshot_round.pop(member_id)
            self._message_snapshot_len[member_key] = self._message_snapshot_len.pop(member_id, 0)

        inbox = self.messages.get(member_key, [])
        snapshot_round = self._message_snapshot_round.get(member_key)

        if create_snapshot and snapshot_round != round_num:
            self._message_snapshot_round[member_key] = round_num
            self._message_snapshot_len[member_key] = len(inbox)
            snapshot_round = round_num

        snapshot_len = self._message_snapshot_len.get(member_key)
        if snapshot_round == round_num and snapshot_len is not None:
            return inbox[:snapshot_len]
        return inbox

    def _consume_message_snapshots(self, round_num: int) -> None:
        """Remove messages that have been surfaced to agents this round."""
        if not self._message_snapshot_round:
            return

        for member_id, snapshot_round in list(self._message_snapshot_round.items()):
            if snapshot_round != round_num:
                continue
            member_key = self._resolve_member_stable_id(member_id) or member_id
            if (
                member_key != member_id
                and member_key not in self._message_snapshot_round
            ):
                self._message_snapshot_round[member_key] = snapshot_round
                self._message_snapshot_len[member_key] = self._message_snapshot_len.get(member_id, 0)
                self._message_snapshot_round.pop(member_id, None)
                self._message_snapshot_len.pop(member_id, None)
            if (
                member_key != member_id
                and member_key not in self.messages
                and member_id in self.messages
            ):
                self.messages[member_key] = self.messages.pop(member_id)

            snapshot_len = int(self._message_snapshot_len.get(member_key, 0) or 0)
            inbox = self.messages.get(member_key, [])
            if snapshot_len > 0:
                snapshot_len = min(snapshot_len, len(inbox))
                remaining = inbox[snapshot_len:]
                if remaining:
                    self.messages[member_key] = remaining
                else:
                    self.messages.pop(member_key, None)
            self._message_snapshot_round.pop(member_key, None)
            self._message_snapshot_len.pop(member_key, None)


