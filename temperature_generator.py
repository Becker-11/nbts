"""Generator module.

This module contains classes/functions used to autogenerate piecewise functions
defining time-dependent temperatures corresponding to an arbitrary set of
simple baking instructions (i.e., heating, cooling, and stable temperature).
"""

from typing import Sequence
import numpy as np
import yaml


class TemperatureGenerator:
    """Temperature generator class.

    This class generates an arbitrary piecewise function defining the time-dependence
    of the temperature during baking.

    Attributes:
        _conditions: List of valid conditions ("stable", "heating", and "cooling").
        _times: List times at each instruction step.
        t_max: Maximum time covered by the generator (h).
        _boundaries: List of generated strings used to dynamically defines the piecewise function's evaluation boundaries.
        _functions: List of generated strings used to dynamically lambda functions called by the piecewise function within each evaluation boundary.
    """

    # valid conditions common to each class instance
    _conditions: list[str] = ["stable", "heating", "cooling"]

    def __init__(
        self,
        instructions: str,
    ) -> None:
        """Initialization.

        Args:
            instructions: Path/name of .yaml file containing the heating instructions.
        """

        # read the yaml file containing the heating instructions
        with open(instructions, "r") as fh:
            self.instructions = yaml.load(fh, Loader=yaml.SafeLoader)

        # make sure the instructions are valid!
        self._check_instructions()

        # useful time containers/constants
        self._times = [step["time (h)"] for step in self.instructions]
        self.t_max = sum(self._times)

        # empty lists to hold relevant information for auto-generating the
        # piecewise temperature function
        self._boundaries = []
        self._functions = []

        # loop over each instruction and populate the empty lists
        for i, step in enumerate(self.instructions):
            # print(instruction)

            # generate the function boundaries
            if i == 0:
                self._boundaries.append(f"(t >= 0) & (t <= {step['time (h)']})")
            else:
                #
                t_1 = sum(self._times[:i])
                t_2 = t_1 + step["time (h)"]

                self._boundaries.append(f"(t > {t_1}) & (t <= {t_2})")

            # generate the call functions
            match step["condition"]:
                case "stable":
                    # lambda function for constant temperature
                    self._functions.append(
                        f"lambda t: 0 * t + {step['temperature (K)']}"
                    )

                case "heating":
                    # time boundaries
                    t_1 = sum(self._times[:i])
                    t_2 = t_1 + step["time (h)"]

                    # temperature boundaries
                    T_1 = self.instructions[i - 1]["temperature (K)"]
                    T_2 = step["temperature (K)"]

                    # calculate the slope/intercept of the linear function
                    slope = (T_2 - T_1) / (t_2 - t_1)
                    intercept = T_2 - slope * t_2

                    # lambda function for a linearly increasing temperature
                    self._functions.append(f"lambda t: {slope} * t + {intercept}")

                case "cooling":
                    # time boundaries
                    t_1 = sum(self._times[:i])
                    t_2 = t_1 + step["time (h)"]

                    # temperature boundaries
                    T_1 = self.instructions[i - 1]["temperature (K)"]
                    T_2 = step["temperature (K)"]

                    # define a "small" number
                    # (needed for finite solutions to the exponential decay constant)
                    epsilon = 0.1  # np.sqrt(np.finfo(float).eps)

                    # calculate exponential "amplitude" and decay constant
                    amplitude = T_1 - T_2
                    constant = -1.0 * np.log(epsilon / amplitude) / (t_2 - t_1)

                    # lambda function for an exponentially decreasing temperature
                    self._functions.append(
                        f"lambda t: {amplitude} * np.exp(-{constant} * (t - {t_1})) + {T_2}"
                    )

    def __call__(
        self,
        t: Sequence[float],
    ) -> Sequence[float]:
        """Piecewise temperature function.

        Piecewise temperature function auto-generated from the baking instructions.

        Args:
            t: Time (s).

        Returns:
            The temperature (K).
        """

        # Pass np to the scope of eval()
        return np.piecewise(
            t,
            [eval(boundary, {"t": t, "np": np}) for boundary in self._boundaries],
            [eval(function, {"t": t, "np": np}) for function in self._functions],
        )



    def _check_instructions(
        self,
    ) -> None:
        """Check that the instructions are valid.

        Perform checks to ensure that the parsed contents of the instructions
        .yaml file adhere to the assumptions used to generate the time-dependent
        temperature function.d
        """

        # make sure the instructions are a list
        assert type(self.instructions) is list

        # check each entry in the instructions
        for step in self.instructions:

            # make sure each step is a dictionary
            assert type(step) is dict

            # make sure each dictionary contains these keys
            assert "temperature (K)" in step
            assert "condition" in step
            assert "time (h)" in step

            # make sure the temperature value is:
            # - a number, finite, and positive
            assert isinstance(
                step["temperature (K)"], (int, float, complex)
            ) and not isinstance(step["temperature (K)"], bool)
            assert np.isfinite(step["temperature (K)"])
            assert step["temperature (K)"] >= 0

            # make sure the time is:
            # - a number, finite, and positive
            assert isinstance(
                step["time (h)"], (int, float, complex)
            ) and not isinstance(step["time (h)"], bool)
            assert np.isfinite(step["time (h)"])
            assert step["time (h)"] >= 0

            # make sure the condition is one of three possibilities
            assert step["condition"] in self._conditions

        # make sure the first "step" is just a constant temperature
        assert self.instructions[0]["condition"] == "stable"
