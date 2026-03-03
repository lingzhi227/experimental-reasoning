"""Context Management Module (CMM) — the agent's structured memory.

Three-level context management:
  L1 (active context, in-prompt): current hypotheses + recent evidence + tactic state
  L2 (compressed summaries, queryable): prior experiment key findings
  L3 (full archive, SQLite): complete provenance graph, on-demand query
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Hypotheses registry
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    confidence REAL NOT NULL DEFAULT 0.5,
    parent_id TEXT REFERENCES hypotheses(id),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Evidence store
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    source_action_id TEXT REFERENCES actions(id),
    content TEXT NOT NULL,
    evidence_type TEXT NOT NULL DEFAULT 'observation',
    created_at REAL NOT NULL
);

-- Evidence ↔ Hypothesis links
CREATE TABLE IF NOT EXISTS evidence_hypothesis (
    evidence_id TEXT NOT NULL REFERENCES evidence(id),
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id),
    relation TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY (evidence_id, hypothesis_id)
);

-- Action log (every tool call / code execution)
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    cycle_id TEXT,
    state_name TEXT NOT NULL,
    tactic_name TEXT,
    input_summary TEXT,
    output_summary TEXT,
    tokens_used INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

-- Prov-O provenance graph
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Observations (raw outputs from actions)
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES actions(id),
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text',
    truncated_summary TEXT,
    created_at REAL NOT NULL
);

-- L2 experiment summaries
CREATE TABLE IF NOT EXISTS experiment_summaries (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    hypothesis_id TEXT REFERENCES hypotheses(id),
    summary_text TEXT NOT NULL,
    key_findings TEXT,
    outcome TEXT,
    created_at REAL NOT NULL
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_parent ON hypotheses(parent_id);
CREATE INDEX IF NOT EXISTS idx_evidence_action ON evidence(source_action_id);
CREATE INDEX IF NOT EXISTS idx_evidence_hyp_hyp ON evidence_hypothesis(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_actions_cycle ON actions(cycle_id);
CREATE INDEX IF NOT EXISTS idx_actions_state ON actions(state_name);
CREATE INDEX IF NOT EXISTS idx_provenance_subject ON provenance(subject_id, subject_type);
CREATE INDEX IF NOT EXISTS idx_provenance_object ON provenance(object_id, object_type);
CREATE INDEX IF NOT EXISTS idx_observations_action ON observations(action_id);
CREATE INDEX IF NOT EXISTS idx_summaries_cycle ON experiment_summaries(cycle_id);
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

HypothesisStatus = Literal["proposed", "testing", "supported", "refuted", "revised"]
EvidenceRelation = Literal["supports", "contradicts", "neutral"]
ProvenancePredicate = Literal[
    "wasGeneratedBy", "wasDerivedFrom", "used", "wasInformedBy"
]


@dataclass
class HypothesisRecord:
    id: str
    statement: str
    status: HypothesisStatus
    confidence: float
    parent_id: str | None
    created_at: float
    updated_at: float


@dataclass
class EvidenceRecord:
    id: str
    source_action_id: str | None
    content: str
    evidence_type: str
    created_at: float


@dataclass
class ActionRecord:
    id: str
    cycle_id: str | None
    state_name: str
    tactic_name: str | None
    input_summary: str | None
    output_summary: str | None
    tokens_used: int
    duration_ms: int
    created_at: float


@dataclass
class ObservationRecord:
    id: str
    action_id: str
    content: str
    content_type: str
    truncated_summary: str | None
    created_at: float


# ---------------------------------------------------------------------------
# L1 Active Context — what goes into the prompt
# ---------------------------------------------------------------------------

@dataclass
class L1Context:
    """Active context assembled for prompt injection."""
    active_hypotheses: list[HypothesisRecord] = field(default_factory=list)
    recent_evidence: list[dict[str, Any]] = field(default_factory=list)
    current_cycle_id: str | None = None
    current_state: str | None = None
    tactic_state: dict[str, Any] = field(default_factory=dict)

    def to_prompt_section(self) -> str:
        """Render L1 context as a prompt section."""
        lines = ["## Active Context"]
        if self.current_state:
            lines.append(f"**Current State**: {self.current_state}")
        if self.current_cycle_id:
            lines.append(f"**Cycle**: {self.current_cycle_id}")

        if self.active_hypotheses:
            lines.append("\n### Active Hypotheses")
            for h in self.active_hypotheses:
                status_icon = {
                    "proposed": "?", "testing": "~",
                    "supported": "+", "refuted": "x", "revised": ">"
                }.get(h.status, "?")
                lines.append(
                    f"- [{status_icon}] **{h.id}** (conf={h.confidence:.2f}): {h.statement}"
                )

        if self.recent_evidence:
            lines.append("\n### Recent Evidence")
            for ev in self.recent_evidence[-5:]:
                rel = ev.get("relation", "?")
                hyp = ev.get("hypothesis_id", "?")
                content = ev.get("content", "")[:200]
                lines.append(f"- [{rel}→{hyp}] {content}")

        if self.tactic_state:
            lines.append("\n### Tactic State")
            for k, v in self.tactic_state.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CMM Database
# ---------------------------------------------------------------------------

class CMMDatabase:
    """Context Management Module — SQLite-backed structured memory."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Path.home() / ".tactical" / "er_context.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def initialize(self) -> None:
        """Create all tables and indexes."""
        cur = self.conn.cursor()
        cur.executescript(SCHEMA_SQL)
        cur.executescript(INDEX_SQL)
        cur.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ── Hypothesis operations ──

    def add_hypothesis(
        self,
        statement: str,
        confidence: float = 0.5,
        parent_id: str | None = None,
        status: HypothesisStatus = "proposed",
    ) -> str:
        hid = f"H-{uuid.uuid4().hex[:8]}"
        now = time.time()
        self.conn.execute(
            """INSERT INTO hypotheses (id, statement, status, confidence, parent_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (hid, statement, status, confidence, parent_id, now, now),
        )
        self.conn.commit()
        return hid

    def update_hypothesis(
        self,
        hypothesis_id: str,
        status: HypothesisStatus | None = None,
        confidence: float | None = None,
    ) -> None:
        updates = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(hypothesis_id)
        self.conn.execute(
            f"UPDATE hypotheses SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisRecord | None:
        row = self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        if not row:
            return None
        return HypothesisRecord(**dict(row))

    def get_active_hypotheses(self) -> list[HypothesisRecord]:
        rows = self.conn.execute(
            """SELECT * FROM hypotheses
            WHERE status IN ('proposed', 'testing')
            ORDER BY confidence DESC, updated_at DESC"""
        ).fetchall()
        return [HypothesisRecord(**dict(r)) for r in rows]

    def get_all_hypotheses(self) -> list[HypothesisRecord]:
        rows = self.conn.execute(
            "SELECT * FROM hypotheses ORDER BY created_at"
        ).fetchall()
        return [HypothesisRecord(**dict(r)) for r in rows]

    # ── Evidence operations ──

    def add_evidence(
        self,
        content: str,
        evidence_type: str = "observation",
        source_action_id: str | None = None,
    ) -> str:
        eid = f"E-{uuid.uuid4().hex[:8]}"
        now = time.time()
        self.conn.execute(
            """INSERT INTO evidence (id, source_action_id, content, evidence_type, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (eid, source_action_id, content, evidence_type, now),
        )
        self.conn.commit()
        return eid

    def link_evidence_hypothesis(
        self,
        evidence_id: str,
        hypothesis_id: str,
        relation: EvidenceRelation,
        strength: float = 0.5,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO evidence_hypothesis
            (evidence_id, hypothesis_id, relation, strength)
            VALUES (?, ?, ?, ?)""",
            (evidence_id, hypothesis_id, relation, strength),
        )
        self.conn.commit()

    def get_evidence_for_hypothesis(self, hypothesis_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.*, eh.relation, eh.strength
            FROM evidence e
            JOIN evidence_hypothesis eh ON e.id = eh.evidence_id
            WHERE eh.hypothesis_id = ?
            ORDER BY e.created_at DESC""",
            (hypothesis_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_evidence(self, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.*, eh.relation, eh.hypothesis_id, eh.strength
            FROM evidence e
            LEFT JOIN evidence_hypothesis eh ON e.id = eh.evidence_id
            ORDER BY e.created_at DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Action log operations ──

    def log_action(
        self,
        state_name: str,
        cycle_id: str | None = None,
        tactic_name: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        tokens_used: int = 0,
        duration_ms: int = 0,
    ) -> str:
        aid = f"A-{uuid.uuid4().hex[:8]}"
        now = time.time()
        self.conn.execute(
            """INSERT INTO actions
            (id, cycle_id, state_name, tactic_name, input_summary, output_summary,
             tokens_used, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (aid, cycle_id, state_name, tactic_name, input_summary,
             output_summary, tokens_used, duration_ms, now),
        )
        self.conn.commit()
        return aid

    def get_cycle_actions(self, cycle_id: str) -> list[ActionRecord]:
        rows = self.conn.execute(
            "SELECT * FROM actions WHERE cycle_id = ? ORDER BY created_at",
            (cycle_id,),
        ).fetchall()
        return [ActionRecord(**dict(r)) for r in rows]

    # ── Observation operations ──

    def add_observation(
        self,
        action_id: str,
        content: str,
        content_type: str = "text",
        truncated_summary: str | None = None,
    ) -> str:
        oid = f"O-{uuid.uuid4().hex[:8]}"
        now = time.time()
        self.conn.execute(
            """INSERT INTO observations
            (id, action_id, content, content_type, truncated_summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (oid, action_id, content, content_type, truncated_summary, now),
        )
        self.conn.commit()
        return oid

    def get_action_observations(self, action_id: str) -> list[ObservationRecord]:
        rows = self.conn.execute(
            "SELECT * FROM observations WHERE action_id = ? ORDER BY created_at",
            (action_id,),
        ).fetchall()
        return [ObservationRecord(**dict(r)) for r in rows]

    # ── Provenance graph ──

    def add_provenance(
        self,
        subject_id: str,
        subject_type: str,
        predicate: ProvenancePredicate,
        object_id: str,
        object_type: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO provenance
            (subject_id, subject_type, predicate, object_id, object_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (subject_id, subject_type, predicate, object_id, object_type, time.time()),
        )
        self.conn.commit()

    def trace_provenance(
        self, entity_id: str, direction: str = "backward"
    ) -> list[dict]:
        """Trace provenance chain for an entity.

        direction='backward': what was this derived from?
        direction='forward': what was derived from this?
        """
        if direction == "backward":
            rows = self.conn.execute(
                """SELECT * FROM provenance WHERE subject_id = ?
                ORDER BY created_at DESC""",
                (entity_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM provenance WHERE object_id = ?
                ORDER BY created_at DESC""",
                (entity_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── L2 experiment summaries ──

    def add_experiment_summary(
        self,
        cycle_id: str,
        summary_text: str,
        hypothesis_id: str | None = None,
        key_findings: str | None = None,
        outcome: str | None = None,
    ) -> str:
        sid = f"S-{uuid.uuid4().hex[:8]}"
        self.conn.execute(
            """INSERT INTO experiment_summaries
            (id, cycle_id, hypothesis_id, summary_text, key_findings, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sid, cycle_id, hypothesis_id, summary_text, key_findings, outcome, time.time()),
        )
        self.conn.commit()
        return sid

    def get_experiment_summaries(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM experiment_summaries
            ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── L1 Context assembly ──

    def assemble_l1_context(
        self,
        current_state: str | None = None,
        cycle_id: str | None = None,
        max_hypotheses: int = 5,
        max_evidence: int = 5,
    ) -> L1Context:
        """Assemble the active context for prompt injection."""
        return L1Context(
            active_hypotheses=self.get_active_hypotheses()[:max_hypotheses],
            recent_evidence=self.get_recent_evidence(limit=max_evidence),
            current_cycle_id=cycle_id,
            current_state=current_state,
        )

    # ── Stats ──

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for table in ["hypotheses", "evidence", "actions", "observations",
                       "provenance", "experiment_summaries"]:
            row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            result[f"total_{table}"] = row[0]
        # Hypothesis status breakdown
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM hypotheses GROUP BY status"
        ).fetchall()
        result["hypotheses_by_status"] = {r["status"]: r["cnt"] for r in rows}
        return result
