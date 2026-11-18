"""Teacher-access guard for continual distillation.

Protocol (CDTDD / DIL): at stage t the student may query ONLY teacher T_t, and only during
stage t. After the stage T_t must be gone: no live module, no weights in the student, and
(in strict mode) no stored outputs of T_t used at later stages.

The guard enforces and records this at runtime:
  * load(si, stage, dev) loads T_t, refusing any teacher that does not belong to stage si
                        and refusing to load while another teacher is still held;
  * release()           drops the guard's reference, runs GC and verifies through a weakref
                        that the teacher object is really freed (raises if something
                        still holds it, e.g. a hook, a router or a cached closure);
  * note_*()            per-stage audit of what the student was trained from;
  * write()             teacher_access_audit.json in the run directory.

Strict mode (`protocol.strict_teacher_access: true`) additionally forbids `teacher_echo`,
because echoes are stored soft predictions of past teachers that keep supervising later
stages. With strict mode off, echo use is allowed but logged as past-teacher output use."""
import gc
import weakref
from pathlib import Path

from .common import write_json
from .teachers import load_teacher


class TeacherAccessError(RuntimeError):
    pass


class TeacherAccessGuard:
    def __init__(self, cfg, stages, out_dir):
        self.cfg, self.stages = cfg, stages
        self.strict = bool(cfg["protocol"].get("strict_teacher_access", False))
        self.out = Path(out_dir)
        self._held, self._ref, self._stage = None, None, None
        self.audit = {"strict_teacher_access": self.strict, "stages": []}
        if self.strict and cfg["components"]["teacher_echo"]:
            raise TeacherAccessError("strict_teacher_access forbids components.teacher_echo: echoes are stored "
                                     "outputs of past teachers. Disable teacher_echo or strict mode.")

    # ------------------------------------------------------------------ load / release
    def load(self, si, stage, device):
        allowed = self.stages[si]["teacher"]
        if stage["teacher"] != allowed:
            raise TeacherAccessError(f"stage {si + 1} may only use teacher '{allowed}', got '{stage['teacher']}'")
        if self._ref is not None and self._ref() is not None:
            raise TeacherAccessError(f"teacher of stage '{self._stage}' is still alive while loading '{allowed}'")
        teacher = load_teacher(self.cfg, allowed, device)
        for p in teacher.parameters():
            if p.requires_grad:
                raise TeacherAccessError("teacher parameters must be frozen")
        self._ref, self._stage = weakref.ref(teacher), stage["name"]
        self._cur = {"stage": stage["name"], "index": si + 1, "teacher_loaded": allowed,
                     "other_teachers_loaded": [], "gt_labels_in_loss": False,
                     "past_teacher_echo_batches": {}, "router_calibration": None, "teacher_released": None}
        return teacher

    def release(self):
        """Call after the caller has dropped every reference it owns (`del teacher`, routers, ...)."""
        if self._ref is None:
            if getattr(self, "_cur", None) is not None:
                self._cur["teacher_released"] = "not loaded"
                self._finish()
            return
        gc.collect()
        alive = self._ref() is not None
        self._cur["teacher_released"] = not alive
        self._finish()
        if alive:
            refs = [type(r).__name__ for r in gc.get_referrers(self._ref())][:5]
            raise TeacherAccessError(f"teacher of stage '{self._stage}' is still referenced after release "
                                     f"(referrers: {refs}); the student could keep querying it")
        self._ref, self._stage = None, None

    def _finish(self):
        self.audit["stages"].append(self._cur)
        self._cur = None

    def begin_without_teacher(self, si, stage):
        self._cur = {"stage": stage["name"], "index": si + 1, "teacher_loaded": None,
                     "other_teachers_loaded": [], "gt_labels_in_loss": False,
                     "past_teacher_echo_batches": {}, "router_calibration": None, "teacher_released": None}

    # ------------------------------------------------------------------ audit notes
    def note_labels(self, used):
        self._cur["gt_labels_in_loss"] = self._cur["gt_labels_in_loss"] or bool(used)

    def note_echo(self, source_stages):
        if self.strict and source_stages:
            raise TeacherAccessError(f"strict mode: stored outputs of past teachers used: {source_stages}")
        for s in source_stages:
            self._cur["past_teacher_echo_batches"][s] = self._cur["past_teacher_echo_batches"].get(s, 0) + 1

    def note_router(self, source):
        self._cur["router_calibration"] = source

    def write(self):
        a = self.audit
        a["summary"] = {
            "only_current_teacher_each_stage": all(s["teacher_loaded"] in (None, self.stages[s["index"] - 1]["teacher"])
                                                   and not s["other_teachers_loaded"] for s in a["stages"]),
            "all_teachers_released": all(s["teacher_released"] in (True, "not loaded") for s in a["stages"]),
            "gt_labels_used_in_any_loss": any(s["gt_labels_in_loss"] for s in a["stages"]),
            "past_teacher_outputs_used": any(s["past_teacher_echo_batches"] for s in a["stages"]),
        }
        write_json(self.out / "teacher_access_audit.json", a)
        return a["summary"]
