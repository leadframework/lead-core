"""Custom parameter types that can be defined on `lembas.Case` instances."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from lembas.case import Case


# Create alias for built-in type function so that we can re-assign a variable with that name
_builtin_type = type


class _NoDefault:
    """Used as a sentinel to indicate lack of a default value for an `InputAttribute`."""


class InputParameter:
    """An input parameter which can be defined for a ``Case``, which determines case-specific values.

    The parameter is of a defined type and type conversion will be performed when setting the value,
    if appropriate.

    Args:
        type: The parameter type, e.g. ``float``, ``str``, etc.
        default: The default value. If the type is not set, the type of the default will be used.
        min: The minimum value for the parameter (if it is a float).
        max: The maximum value for the parameter (if it is a float).
        control: Set as True to indicate a runtime control parameter.
            These parameters will be excluded from the `case.inputs` dictionary.

    """

    _name: str

    def __init__(
        self,
        *,
        type: type | None = None,
        default: Any = _NoDefault,
        min: float | None = None,
        max: float | None = None,
        control: bool = False,
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
        self._control = control

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

    @cached_property
    def include_in_inputs_dict(self) -> bool:
        """If True, include the parameter in the `case.inputs` dictionary."""
        return not self._control
