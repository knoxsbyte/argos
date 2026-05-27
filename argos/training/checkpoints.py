"""
argos.training.checkpoints — Checkpoint registry for trained ARGOS policies.

Records checkpoint metadata (path, epoch, metrics, model type) in a JSON
index file alongside the saved weights so you can always answer:
  - Which checkpoint had the highest success rate on make_bed?
  - Which of these two runs should I deploy?
  - How many checkpoints am I storing and which ones can I prune?

Usage:
    registry = CheckpointRegistry("data/models")
    rec = registry.register(path, epoch=5, task_type="sweep_floor",
                            model_type="openvla", metrics={"success_rate": 0.88})
    best = registry.best(task_type="sweep_floor")
    removed = registry.prune(keep_top=3)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEX_FILE = "checkpoints.json"


@dataclass
class CheckpointRecord:
    """Metadata for one saved model checkpoint."""

    checkpoint_id: str
    path: str                    # path to .pt / .safetensors file
    epoch: int
    task_type: str
    model_type: str              # openvla | diffusion | act
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    is_best: bool = False
    notes: str = ""

    def metric(self, key: str) -> float | None:
        return self.metrics.get(key)

    def created_iso(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M")


class CheckpointRegistry:
    """Persists and queries checkpoint metadata for trained ARGOS policies.

    All records live in ``<registry_dir>/checkpoints.json``.  The file is
    written atomically on every mutation so a crash never corrupts it.

    Parameters
    ----------
    registry_dir:
        Directory that contains (or will contain) ``checkpoints.json``.
        Typically ``data/models/``.
    """

    def __init__(self, registry_dir: Path | str = "data/models") -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / _INDEX_FILE
        self._records: dict[str, CheckpointRecord] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            raw = json.loads(self._index_path.read_text())
            for rec in raw.get("checkpoints", []):
                r = CheckpointRecord(**rec)
                self._records[r.checkpoint_id] = r
            logger.debug("CheckpointRegistry: loaded %d record(s)", len(self._records))
        except Exception as exc:
            logger.warning("CheckpointRegistry: could not load index: %s", exc)

    def _save(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        try:
            payload = {"checkpoints": [asdict(r) for r in self._records.values()]}
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._index_path)
        except Exception as exc:
            logger.warning("CheckpointRegistry: could not save index: %s", exc)
            tmp.unlink(missing_ok=True)

    # ── write operations ──────────────────────────────────────────────────────

    def register(
        self,
        path: Path | str,
        epoch: int,
        task_type: str,
        model_type: str,
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> CheckpointRecord:
        """Register a checkpoint and persist the updated index.

        A short deterministic ID is derived from the path and epoch so the
        same file is never registered twice.
        """
        digest = hashlib.md5(f"{path}:{epoch}".encode()).hexdigest()[:6]
        cid = f"ckpt-{model_type}-ep{epoch:03d}-{digest}"
        rec = CheckpointRecord(
            checkpoint_id=cid,
            path=str(path),
            epoch=epoch,
            task_type=task_type,
            model_type=model_type,
            metrics=metrics or {},
            notes=notes,
        )
        self._records[cid] = rec
        self._save()
        logger.info(
            "Checkpoint registered: %s  epoch=%d  task=%s  metrics=%s",
            cid, epoch, task_type, metrics,
        )
        return rec

    def update_metrics(self, checkpoint_id: str, metrics: dict[str, float]) -> None:
        """Merge new metric values into an existing record (e.g. after evaluation)."""
        rec = self._records.get(checkpoint_id)
        if rec is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")
        rec.metrics.update(metrics)
        self._save()

    def mark_best(
        self,
        checkpoint_id: str,
        metric_key: str = "success_rate",
    ) -> None:
        """Pin one checkpoint as best; clears the flag on all others."""
        if checkpoint_id not in self._records:
            raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")
        for r in self._records.values():
            r.is_best = r.checkpoint_id == checkpoint_id
        self._save()
        logger.info("Checkpoint %s pinned as best (metric=%s)", checkpoint_id, metric_key)

    def delete(self, checkpoint_id: str, delete_file: bool = False) -> None:
        """Remove a checkpoint record, optionally deleting the weights file."""
        rec = self._records.pop(checkpoint_id, None)
        if rec is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")
        if delete_file:
            p = Path(rec.path)
            if p.exists():
                p.unlink()
                logger.info("Deleted weights file: %s", p)
        self._save()

    # ── read operations ───────────────────────────────────────────────────────

    def get(self, checkpoint_id: str) -> CheckpointRecord | None:
        return self._records.get(checkpoint_id)

    def list_all(
        self,
        task_type: str | None = None,
        model_type: str | None = None,
    ) -> list[CheckpointRecord]:
        """Return all records, newest first, with optional filters."""
        records = list(self._records.values())
        if task_type:
            records = [r for r in records if r.task_type == task_type]
        if model_type:
            records = [r for r in records if r.model_type == model_type]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def best(
        self,
        task_type: str | None = None,
        metric_key: str = "success_rate",
        higher_is_better: bool = True,
    ) -> CheckpointRecord | None:
        """Return the checkpoint with the best value for *metric_key*.

        If a record is pinned via :meth:`mark_best` it is returned
        immediately without recomputing the ranking.
        """
        candidates = self.list_all(task_type=task_type)
        pinned = next((r for r in candidates if r.is_best), None)
        if pinned:
            return pinned
        scored = [r for r in candidates if metric_key in r.metrics]
        if not scored:
            return None
        return (max if higher_is_better else min)(
            scored, key=lambda r: r.metrics[metric_key]
        )

    def compare(self, id_a: str, id_b: str) -> dict[str, dict[str, float | None]]:
        """Side-by-side metric comparison of two checkpoints.

        Returns ``{metric_key: {"a": value_a, "b": value_b, "delta": b-a}}``.
        """
        ra = self._records.get(id_a)
        rb = self._records.get(id_b)
        if ra is None or rb is None:
            missing = id_a if ra is None else id_b
            raise KeyError(f"Checkpoint not found: {missing!r}")
        all_keys = sorted(set(ra.metrics) | set(rb.metrics))
        result: dict[str, dict[str, float | None]] = {}
        for k in all_keys:
            va = ra.metrics.get(k)
            vb = rb.metrics.get(k)
            delta = (vb - va) if (va is not None and vb is not None) else None
            result[k] = {"a": va, "b": vb, "delta": delta}
        return result

    def prune(
        self,
        keep_top: int = 5,
        metric_key: str = "success_rate",
        higher_is_better: bool = True,
        delete_files: bool = False,
    ) -> list[str]:
        """Remove all but the top-*keep_top* checkpoints by *metric_key*.

        Checkpoints pinned via :meth:`mark_best` are never pruned regardless
        of their rank.  Records without the target metric are also kept.

        Returns the list of removed checkpoint IDs.
        """
        scored = sorted(
            [r for r in self._records.values() if metric_key in r.metrics],
            key=lambda r: r.metrics[metric_key],
            reverse=higher_is_better,
        )
        to_remove = [r for r in scored[keep_top:] if not r.is_best]
        removed: list[str] = []
        for r in to_remove:
            if delete_files:
                p = Path(r.path)
                if p.exists():
                    p.unlink()
            self._records.pop(r.checkpoint_id, None)
            removed.append(r.checkpoint_id)

        if removed:
            self._save()
            logger.info(
                "Pruned %d checkpoint(s); kept top-%d by %s",
                len(removed), keep_top, metric_key,
            )
        return removed

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"<CheckpointRegistry dir={self._dir} records={len(self)}>"
