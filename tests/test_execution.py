import pytest

from rstar_deepthink.tools.python_tool import (
    execute_code_directly,
    execute_code_in_subprocess,
)


@pytest.mark.parametrize("execute", [execute_code_directly, execute_code_in_subprocess])
def test_candidate_execution(execute):
    inputs = [[[1, 2]], [[3, 4]]]
    error, passed, outputs = execute("def solve(I): return I", inputs, inputs)
    assert not error
    assert passed
    assert outputs == inputs


@pytest.mark.parametrize("execute", [execute_code_directly, execute_code_in_subprocess])
def test_bad_candidate_does_not_abort_the_run(execute):
    error, passed, _ = execute(
        "def solve(I): raise ValueError('bad candidate')", [[[1]]], [[[1]]]
    )
    assert error
    assert not passed
