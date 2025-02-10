from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import toml

from lembas import Case
from lembas import CaseList
from lembas import InputParameter
from lembas import step


class MyCase(Case):
    my_param = InputParameter(type=float, min=2.0, max=5.0)
    param_with_default = InputParameter(default=10.0)
    required_param = InputParameter(type=float)
    control_param = InputParameter(default=1.0, control=True)
    control_param_true = InputParameter(default=True, control=True)
    control_param_false = InputParameter(default=False, control=True)

    # Attributes used to check whether steps have been run
    first_step_has_been_run = False
    second_step_has_been_run = False
    step_has_been_triggered_by_string = False
    step_has_been_triggered_by_string_not = False

    @step(condition=lambda self: self.my_param > 4, requires="second_step")
    def change_param_with_default(self) -> None:
        self.param_with_default = 5.0

    @step(requires="first_step")
    def second_step(self) -> None:
        """Set has_been_run to True."""
        if not self.first_step_has_been_run:
            raise RuntimeError  # pragma: no cover
        self.second_step_has_been_run = True

    @step
    def first_step(self) -> None:
        self.first_step_has_been_run = True

    @step(condition="control_param_true")
    def triggered_by_string(self) -> None:
        self.step_has_been_triggered_by_string = True

    @step(condition="not control_param_false")
    def triggered_by_string_not(self) -> None:
        self.step_has_been_triggered_by_string_not = True


@pytest.fixture()
def case() -> MyCase:
    return MyCase(my_param=3.0)


def test_case_parameter_default(case: MyCase) -> None:
    assert case.param_with_default == pytest.approx(10.0)


def test_case_parameter_required_raises_exception(case: MyCase) -> None:
    """A parameter with no default that is not set raises an exception when accessed."""
    with pytest.raises(AttributeError):
        _ = case.required_param


def test_case_parameter_type_required() -> None:
    with pytest.raises(TypeError):
        InputParameter()


@pytest.mark.parametrize("input_value", [3.0, 3, "3", "3.0"])
def test_case_parameter_type_conversion(case: MyCase, input_value: Any) -> None:
    """The value is coerced to the proper type."""
    assert case.my_param == pytest.approx(3.0)


@pytest.mark.parametrize("input_value", [1.0, 6.0])
def test_case_parameter_bounds_raises_exception(
    case: MyCase, input_value: float
) -> None:
    """An exception is raised when attempting to set the value out of bounds."""
    with pytest.raises(ValueError):
        case.my_param = input_value


def test_case_step_docstring(case: MyCase) -> None:
    """The docstring is replaced on a wrapped @step method."""
    assert case.second_step.__doc__ == "Set has_been_run to True."


def test_case_step_name(case: MyCase) -> None:
    assert case.change_param_with_default.name == "change_param_with_default"


def test_case_step_condition_is_not_met(case: MyCase) -> None:
    """If we don't set the case.my_param, the step shouldn't run."""
    case.change_param_with_default()
    assert case.param_with_default == pytest.approx(10.0)


def test_case_step_condition_is_met(case: MyCase) -> None:
    """If we do set the case.my_param explicitly, the step should run."""
    case.my_param = 5.0
    case.change_param_with_default()
    assert case.param_with_default == pytest.approx(5.0)


def test_case_step_string_condition(case: MyCase) -> None:
    """The step with a simple string condition is run."""
    case.run()
    assert case.step_has_been_triggered_by_string


def test_case_step_string_condition_not(case: MyCase) -> None:
    """The step with a string "not" condition is run."""
    case.run()
    assert case.step_has_been_triggered_by_string_not


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param("is some_attribute", id="only-not-allowed"),
        pytest.param("is not some_attribute", id="max-two-words"),
        pytest.param("", id="at-least-one-word"),
    ],
)
def test_string_condition_invalid_raises_exception(condition: str) -> None:
    """The string must be either an attribute name, or `not attribute_name`."""
    with pytest.raises(ValueError):

        class MyClass(Case):
            @step(condition=condition)
            def anything(self) -> None: ...  # pragma: no cover


def test_case_steps_order(case: MyCase) -> None:
    expected_steps = ["first_step", "second_step", "change_param_with_default"]
    # Extract the step names if they are expected (drop ones without a requires)
    step_names = [
        step.name for step in case._sorted_steps if step.name in expected_steps
    ]
    assert step_names == expected_steps


def test_casehandler_full_name(case: MyCase) -> None:
    assert case.fully_resolved_name == "test_case_handler.MyCase"


def test_case_inputs_dict(case: MyCase) -> None:
    case.required_param = 4.0
    assert case.inputs == {
        "my_param": 3.0,
        "param_with_default": 10.0,
        "required_param": 4.0,
    }


def test_case_lembas_toml(case: MyCase, tmp_path: Path) -> None:
    case.case_dir = tmp_path
    case.required_param = 4.0
    assert case.case_dir == tmp_path
    case._write_lembas_file()
    assert (tmp_path / "lembas" / "case.toml").exists()
    with (tmp_path / "lembas" / "case.toml").open("r") as fp:
        data = toml.load(fp)
    assert data == {
        "lembas": {"inputs": case.inputs, "case-handler": case.fully_resolved_name}
    }


def test_case_relative_case_dir_is_absolute(case: MyCase, tmp_path: Path) -> None:
    """If the case_dir is not relative to CWD, the relative_case_dir is an absolute path."""
    case.case_dir = Path("/some/non-relative-path")
    assert case.relative_case_dir == Path("/some/non-relative-path")


def test_case_relative_case_dir_is_relative(case: MyCase, tmp_path: Path) -> None:
    """If the case_dir is relative to CWD, the relative_case_dir is relative."""
    case.case_dir = Path.cwd() / "some/relative-path"
    assert case.relative_case_dir == Path("some/relative-path")


@pytest.fixture()
def case_list() -> CaseList:
    return CaseList()


class TestCaseList:
    """Tests for the `lembas.core.CaseList` class."""

    def test_add_case_returns_case(self, case_list: CaseList, case: MyCase) -> None:
        added_case = case_list.add(case)
        assert added_case is case

    def test_add_case_in(self, case_list: CaseList, case: MyCase) -> None:
        case_list.add(case)
        assert case in case_list

    def test_run_all_cases(self, case_list: CaseList, case: MyCase) -> None:
        case_list.add(case)
        assert not case.second_step_has_been_run
        case_list.run_all()
        assert case.second_step_has_been_run

    def test_build_from_iterable(self) -> None:
        case_list = CaseList(MyCase(required_param=i) for i in range(10))
        assert len(case_list) == 10

    def test_is_iterable(self) -> None:
        case_list = CaseList(MyCase(required_param=i) for i in range(10))
        assert [case.required_param for case in case_list] == list(range(10))

    def test_add_param_sweep(self, case_list: CaseList) -> None:
        case_list.add_cases_by_parameter_sweep(
            MyCase,
            required_param=1.0,
            my_param=range(2, 6),
            param_with_default=range(5),
        )
        assert len(case_list) == 20
