"""Hypothesis Manager — lifecycle management for scientific hypotheses.

Manages the full hypothesis lifecycle:
  proposed → testing → supported/refuted → revised

Supports parent-child relationships for hypothesis revision chains,
confidence tracking based on evidence accumulation, and priority
queue ordering for experiment planning.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from .cmm import CMMDatabase, EvidenceRelation, HypothesisRecord, HypothesisStatus


@dataclass(order=True)
class PrioritizedHypothesis:
    """Wrapper for priority queue ordering (higher confidence = higher priority)."""
    priority: float
    hypothesis_id: str = field(compare=False)

    @classmethod
    def from_record(cls, rec: HypothesisRecord) -> PrioritizedHypothesis:
        # Negate confidence for max-heap behavior with heapq (min-heap)
        return cls(priority=-rec.confidence, hypothesis_id=rec.id)


class HypothesisManager:
    """Manages hypothesis lifecycle with evidence-based updates.

    Operations:
      propose()  → create a new hypothesis
      test()     → mark as under testing
      add_evidence() → link evidence and update confidence
      support()  → mark hypothesis as supported
      refute()   → mark hypothesis as refuted
      revise()   → create a revised hypothesis (child of refuted one)
      next_to_test() → get highest priority untested hypothesis
    """

    def __init__(self, cmm: CMMDatabase) -> None:
        self.cmm = cmm
        self._queue: list[PrioritizedHypothesis] = []
        self._rebuild_queue()

    def _rebuild_queue(self) -> None:
        """Rebuild the priority queue from active hypotheses."""
        self._queue.clear()
        for h in self.cmm.get_active_hypotheses():
            heapq.heappush(self._queue, PrioritizedHypothesis.from_record(h))

    # ── Lifecycle operations ──

    def propose(
        self,
        statement: str,
        confidence: float = 0.5,
        parent_id: str | None = None,
    ) -> str:
        """Create a new hypothesis in 'proposed' status."""
        hid = self.cmm.add_hypothesis(
            statement=statement,
            confidence=confidence,
            parent_id=parent_id,
            status="proposed",
        )
        rec = self.cmm.get_hypothesis(hid)
        if rec:
            heapq.heappush(self._queue, PrioritizedHypothesis.from_record(rec))
        return hid

    def test(self, hypothesis_id: str) -> None:
        """Transition hypothesis to 'testing' status."""
        self.cmm.update_hypothesis(hypothesis_id, status="testing")

    def support(self, hypothesis_id: str, confidence: float | None = None) -> None:
        """Mark hypothesis as supported."""
        self.cmm.update_hypothesis(
            hypothesis_id,
            status="supported",
            confidence=confidence,
        )

    def refute(self, hypothesis_id: str, confidence: float | None = None) -> None:
        """Mark hypothesis as refuted."""
        self.cmm.update_hypothesis(
            hypothesis_id,
            status="refuted",
            confidence=confidence or 0.0,
        )

    def revise(
        self,
        original_id: str,
        new_statement: str,
        confidence: float = 0.5,
    ) -> str:
        """Create a revised hypothesis derived from an existing one.

        Marks the original as 'revised' and creates a new child hypothesis.
        """
        self.cmm.update_hypothesis(original_id, status="revised")
        new_id = self.propose(
            statement=new_statement,
            confidence=confidence,
            parent_id=original_id,
        )
        # Record provenance: new hypothesis wasDerivedFrom original
        self.cmm.add_provenance(
            subject_id=new_id,
            subject_type="hypothesis",
            predicate="wasDerivedFrom",
            object_id=original_id,
            object_type="hypothesis",
        )
        return new_id

    # ── Evidence integration ──

    def add_evidence(
        self,
        hypothesis_id: str,
        content: str,
        relation: EvidenceRelation,
        strength: float = 0.5,
        evidence_type: str = "observation",
        source_action_id: str | None = None,
    ) -> str:
        """Add evidence linked to a hypothesis and update confidence.

        Returns the evidence ID.
        """
        eid = self.cmm.add_evidence(
            content=content,
            evidence_type=evidence_type,
            source_action_id=source_action_id,
        )
        self.cmm.link_evidence_hypothesis(
            evidence_id=eid,
            hypothesis_id=hypothesis_id,
            relation=relation,
            strength=strength,
        )
        # Record provenance
        if source_action_id:
            self.cmm.add_provenance(
                subject_id=eid,
                subject_type="evidence",
                predicate="wasGeneratedBy",
                object_id=source_action_id,
                object_type="action",
            )

        # Update confidence based on accumulated evidence
        self._update_confidence(hypothesis_id)
        return eid

    def _update_confidence(self, hypothesis_id: str) -> None:
        """Recalculate hypothesis confidence from all linked evidence.

        Uses a Bayesian-inspired update:
          - supporting evidence increases confidence
          - contradicting evidence decreases confidence
          - neutral evidence has minimal effect
        Each piece of evidence shifts confidence by strength * direction_factor.
        """
        evidence_list = self.cmm.get_evidence_for_hypothesis(hypothesis_id)
        if not evidence_list:
            return

        # Start from neutral prior
        confidence = 0.5
        for ev in evidence_list:
            strength = ev.get("strength", 0.5)
            relation = ev.get("relation", "neutral")
            if relation == "supports":
                # Move toward 1.0
                confidence += (1.0 - confidence) * strength * 0.3
            elif relation == "contradicts":
                # Move toward 0.0
                confidence -= confidence * strength * 0.3
            # neutral: minimal shift
            else:
                confidence += (0.5 - confidence) * strength * 0.05

        confidence = max(0.0, min(1.0, confidence))
        self.cmm.update_hypothesis(hypothesis_id, confidence=confidence)

    # ── Query operations ──

    def next_to_test(self) -> HypothesisRecord | None:
        """Get the highest-priority hypothesis that needs testing."""
        # Clean stale entries from the queue
        while self._queue:
            item = self._queue[0]
            rec = self.cmm.get_hypothesis(item.hypothesis_id)
            if rec and rec.status in ("proposed", "testing"):
                return rec
            heapq.heappop(self._queue)
        return None

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisRecord | None:
        return self.cmm.get_hypothesis(hypothesis_id)

    def get_active(self) -> list[HypothesisRecord]:
        return self.cmm.get_active_hypotheses()

    def get_all(self) -> list[HypothesisRecord]:
        return self.cmm.get_all_hypotheses()

    def get_evidence(self, hypothesis_id: str) -> list[dict]:
        return self.cmm.get_evidence_for_hypothesis(hypothesis_id)

    def get_revision_chain(self, hypothesis_id: str) -> list[HypothesisRecord]:
        """Trace the full revision chain back to the root hypothesis."""
        chain = []
        current_id: str | None = hypothesis_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            rec = self.cmm.get_hypothesis(current_id)
            if rec is None:
                break
            chain.append(rec)
            current_id = rec.parent_id
        chain.reverse()
        return chain

    def summary(self) -> dict[str, Any]:
        """Summary of hypothesis state for logging/display."""
        all_h = self.get_all()
        by_status: dict[str, int] = {}
        for h in all_h:
            by_status[h.status] = by_status.get(h.status, 0) + 1
        return {
            "total": len(all_h),
            "by_status": by_status,
            "active_count": len(self.get_active()),
            "next_to_test": (self.next_to_test() or HypothesisRecord(
                id="none", statement="", status="proposed",
                confidence=0, parent_id=None, created_at=0, updated_at=0
            )).id,
        }
