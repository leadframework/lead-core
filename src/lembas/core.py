from __future__ import annotations

import itertools
import types
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from functools import WRAPPER_ASSIGNMENTS
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar


class _NoDefault:
    """Used as a sentinel to indicate lack of a default value for an `InputAttribute`."""


_builtin_type = type


class InputParameter:
    _name: str
    _type: type | None
    _min_value: float | None
    _max_value: float | None

    def __init__(
        self,
        *,
        type: type | None = None,
        default: Any = _NoDefault,
        min: float | None = None,
        max: float | None = None,
    ):
        if type is not None:
            self._type = type
        elif default != _NoDefault:
            self._type = _builtin_type(default)
        else:
            raise TypeError(
                "An 'InputParameter' must either explicitly set the type, or have a default value to infer it from."
            )

        self._default = default
        self._min_value = min
        self._max_value = max

    def __set_name__(self, owner: type[Case], name: str) -> None:
        self._name = name

    def __set__(self, instance: object, value: Any) -> None:
        if self._type is not None:
            value = self._type(value)

        if self._min_value is not None and value < self._min_value:
            raise ValueError(
                "Specified value less than minimum for attribute "
                f"'{self._name}': ({value} < {self._min_value})"
            )

        if self._max_value is not None and value > self._max_value:
            raise ValueError(
                "Specified value exceeds maximum for attribute "
                f"'{self._name}': ({value} > {self._max_value})"
            )

        instance.__dict__[self._name] = value

    def __get__(self, instance: Case, owner: type[Case]) -> Any:
        """Retrieve the attribute from the instance dictionary.

        If the value hasn't been set, we attempt to return a default, unless there is no default,
        in which case we raise an AttributeError.

        """
        try:
            return instance.__dict__[self._name]
        except KeyError:
            if self._default == _NoDefault:
                raise AttributeError(
                    f"'{self._name}' attribute of '{owner.__name__}' class "
                    "has no default value and must be specified explicitly"
                )
            return self._default


StepMethod = Callable[["Case"], None]


class CaseStep:
    def __init__(
        self, func: StepMethod, *, condition: Callable[[Any], bool] | None = None
    ):
        self._func = func
        self._condition = condition

    def __call__(self, instance: Case) -> None:
        if self._condition is None or self._condition(instance):
            return self._func(instance)
        return None

    def __get__(self, instance: Any, cls: Any) -> types.MethodType:
        """We need to implement __get__ so that the `CaseStep` can be used as a method.

        Without doing this, Python will treat it as a normal attribute access, rather
        than as a descriptor.

        """
        return types.MethodType(self, instance)


# TODO: I can't figure out how to properly resolve type errors when the argument is `Case`
#       in the decorator definition for condition


def step(condition: Callable[[Any], bool] | None = None) -> Any:
    """A decorator to define steps to be performed when running a `Case`.

    The step should not return a value.

    Args:
        condition: an optional callable which can be used to determine whether the step should run.
            It will receive the `Case` instance as its only argument, and must return a boolean
            which, if True, the step will run. Otherwise, it will be skipped.

    Usage:
        ```
        class MyCase(Case):
            @step(condition=lambda case: case.case_dir.exists())
            def some_analysis_step(self):
                # do something
        ```

    """

    def decorator(f: StepMethod) -> StepMethod:
        new_method = CaseStep(f, condition=condition)
        # This is largely a replica of functools.wraps, which doesn't seem to work
        for attr in WRAPPER_ASSIGNMENTS:
            setattr(new_method, attr, getattr(f, attr))
        return new_method

    return decorator


class Case:
    """Base case for all cases."""

    _steps: ClassVar[list[StepMethod]]

    def __init_subclass__(cls, **kwargs: Any):
        cls._steps = [
            method for method in cls.__dict__.values() if isinstance(method, CaseStep)
        ]

    def __init__(self, **kwargs: Any):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def run(self) -> None:
        """The default behavior is to run all of the methods decorated with `@step`."""
        for step_method in self._steps:
            step_method(self)


TCase = TypeVar("TCase", bound=Case)


class CaseList(Generic[TCase]):
    """A generic collection of `Case` objects, and utility methods to run them."""

    def __init__(self, cases: Iterable[TCase] | None = None):
        self._cases: list[TCase] = list(cases or ())

    def add(self, case: TCase) -> TCase:
        """Add a case to the list:

        Args:
            case: The case to add.

        Returns:
            The case that was added.

        """
        self._cases.append(case)
        return case

    def add_cases_by_parameter_sweep(
        self, case_class: type[TCase], **kwargs: Any
    ) -> None:
        """Add a number of cases by performing a parameter sweep using the Cartesian product.

        Args:
            case_class: The type of case to construct.
            kwargs: Any parameters to pass to the case constructors. If iterable values are provided,
                they will be used when performing the parameter sweep via `itertools.product`.

        """
        # Ensure all kwargs have iterable values by wrapping scalars and strings
        for key, value in kwargs.items():
            if isinstance(value, str) or not isinstance(value, Iterable):
                kwargs[key] = [value]

        for values in itertools.product(*kwargs.values()):
            new_kwargs = {k: v for k, v in zip(kwargs.keys(), values)}
            case = case_class(**new_kwargs)
            self.add(case)

    def run_all(self) -> None:
        """Run all the cases."""
        for case in self._cases:
            case.run()

    def __contains__(self, item: TCase) -> bool:
        return item in self._cases

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[TCase]:
        for case in self._cases:
            yield case
