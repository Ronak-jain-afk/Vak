from src.optim import get_lr_multiplier, resume_rewarm_multiplier


def test_resume_rewarm_caps_cosine_jump():
    # Extending 946 -> 1908 steps under cosine jumps the multiplier 0.10x -> ~0.60x
    sched = get_lr_multiplier(946, 1908, 100, 0.1, schedule="cosine")
    assert sched > 0.5, f"Expected schedule jump above 0.5x, got {sched}"
    first = resume_rewarm_multiplier(sched, 946, 946, 100, 0.1)
    assert abs(first - 0.1) < 0.01, f"First resumed step should start near floor, got {first}"


def test_resume_rewarm_converges_to_schedule():
    sched = 0.6
    mid = resume_rewarm_multiplier(sched, 946 + 49, 946, 100, 0.1)
    assert 0.1 < mid < sched
    after = resume_rewarm_multiplier(sched, 946 + 100, 946, 100, 0.1)
    assert after == sched
    before = resume_rewarm_multiplier(sched, 945, 946, 100, 0.1)
    assert before == sched


def test_resume_rewarm_disabled_with_zero_steps():
    assert resume_rewarm_multiplier(0.6, 946, 946, 0, 0.1) == 0.6
