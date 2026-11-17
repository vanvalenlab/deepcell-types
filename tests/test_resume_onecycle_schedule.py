"""Unit tests for resume_onecycle_schedule (deepcell_types/training/utils.py).

Bug this guards against: `--resume_path` combined with an increased
`--epochs` used to silently discard the new (longer) OneCycleLR schedule.
`torch.optim.lr_scheduler.LRScheduler.load_state_dict` is a raw
`self.__dict__.update(state_dict)`, so `scheduler.load_state_dict(ckpt["scheduler"])`
overwrote the freshly-built scheduler's `total_steps` (and derived
`_schedule_phases`) with the CHECKPOINTED run's (shorter) values — training
then continued to anneal toward the OLD, already-passed total_steps instead
of the new one, and a resumed run that stepped past the old total_steps would
even raise inside OneCycleLR.get_lr().

resume_onecycle_schedule() is a small importable helper factored out of
scripts/train.py's/pretrain.py's --resume_path block specifically so this
adjust logic is unit-testable without running real training.
"""

import pytest
import torch
import torch.nn as nn

from deepcell_types.training.utils import resume_onecycle_schedule

# These tests call scheduler.step() directly (no optimizer.step() in
# between), which is exactly what the checkpoint fast-forward path does.
# The resulting UserWarning is expected noise, same suppression used by
# tests/test_train_loop_smoke.py.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)`:UserWarning"
)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _build_onecycle(total_steps):
    model = _TinyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


class TestResumeOnecycleScheduleUnchangedEpochs:
    def test_matching_total_steps_loads_verbatim(self):
        """--epochs unchanged: behavior is identical to a plain load_state_dict."""
        _, old_scheduler = _build_onecycle(total_steps=50)
        for _ in range(30):
            old_scheduler.step()
        old_state = old_scheduler.state_dict()
        old_last_lr = old_scheduler.get_last_lr()

        _, new_scheduler = _build_onecycle(total_steps=50)
        resume_onecycle_schedule(new_scheduler, old_state)

        assert new_scheduler.total_steps == 50
        assert new_scheduler.last_epoch == 30
        assert new_scheduler.get_last_lr() == old_last_lr


class TestResumeOnecycleScheduleChangedEpochs:
    def test_longer_epochs_rebuilds_total_steps_and_preserves_position(self):
        """--epochs increased: total_steps reflects the NEW run, not the checkpoint."""
        _, old_scheduler = _build_onecycle(total_steps=50)
        for _ in range(30):
            old_scheduler.step()
        old_state = old_scheduler.state_dict()
        assert old_state["total_steps"] == 50
        assert old_state["last_epoch"] == 30

        _, new_scheduler = _build_onecycle(total_steps=200)
        resume_onecycle_schedule(new_scheduler, old_state)

        # total_steps must reflect the NEW (longer) run, not the checkpoint's.
        assert new_scheduler.total_steps == 200
        # LR position (how many steps already trained) is preserved.
        assert new_scheduler.last_epoch == 30

    def test_can_step_past_old_total_steps_without_raising(self):
        """The whole point of the fix: training can continue past the OLD
        total_steps (50) because the schedule now runs to the NEW total_steps
        (200). Before the fix, a blind load_state_dict() would leave
        total_steps=50, and OneCycleLR.get_lr() raises once last_epoch
        exceeds total_steps.
        """
        _, old_scheduler = _build_onecycle(total_steps=50)
        for _ in range(30):
            old_scheduler.step()
        old_state = old_scheduler.state_dict()

        _, new_scheduler = _build_onecycle(total_steps=200)
        resume_onecycle_schedule(new_scheduler, old_state)

        # Step from 30 up to 150 total steps — well past the checkpoint's
        # old total_steps=50, but comfortably inside the new total_steps=200.
        for _ in range(120):
            new_scheduler.step()
        assert new_scheduler.last_epoch == 150
        # No exception raised means the rebuilt schedule is in effect.

    def test_shorter_epochs_that_still_covers_completed_steps_rebuilds(self):
        """A smaller-but-still-sufficient new total_steps also rebuilds cleanly."""
        _, old_scheduler = _build_onecycle(total_steps=100)
        for _ in range(10):
            old_scheduler.step()
        old_state = old_scheduler.state_dict()

        _, new_scheduler = _build_onecycle(total_steps=40)
        resume_onecycle_schedule(new_scheduler, old_state)

        assert new_scheduler.total_steps == 40
        assert new_scheduler.last_epoch == 10

    def test_resumed_steps_exceeding_new_total_steps_raises_clear_error(self):
        """Guard: if the checkpoint is already further along than the new
        --epochs allows, fail loudly instead of corrupting the schedule.
        """
        _, old_scheduler = _build_onecycle(total_steps=100)
        for _ in range(80):
            old_scheduler.step()
        old_state = old_scheduler.state_dict()

        _, new_scheduler = _build_onecycle(total_steps=50)
        with pytest.raises(ValueError, match="already completed"):
            resume_onecycle_schedule(new_scheduler, old_state)
