"""An example using `lembas` to perform a parameter sweep for a single flat planing plate.

Each case is run with `planingfsi` and is characterized by the Froude number
(flow speed) and angle of attack.

"""

from __future__ import annotations

import shutil
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import NamedTuple

from planingfsi.dictionary import load_dict_from_file

from lembas import Case
from lembas import InputParameter
from lembas import result
from lembas import step


class PlaningPlateResults(NamedTuple):
    drag: float
    lift: float
    moment: float


class PlaningPlateCase(Case):
    froude_num = InputParameter(type=float, min=0.2, max=3.0)
    angle_of_attack = InputParameter(type=float)

    @property
    def case_dir(self) -> Path:
        """The directory in which to run the case."""
        return Path(
            Path.cwd(),
            "cases",
            f"Fr={self.froude_num:0.2f}_AOA={self.angle_of_attack:0.2f}",
        )

    @result("drag", "lift", "moment")
    def load_results(self) -> PlaningPlateResults:
        """Load results from files and return."""
        # Find the latest time directory to load results from
        results_dirs = sorted(
            self.case_dir.glob("[0-9]*"), key=lambda d: int(d.name), reverse=True
        )
        results_dir = results_dirs[0]
        results_dict = load_dict_from_file(results_dir / "forces_total.txt")
        return PlaningPlateResults(
            drag=results_dict["Drag"],
            lift=results_dict["Lift"],
            moment=results_dict["Moment"],
        )

    @cached_property
    def inputs_dict(self) -> dict[str, Any]:
        """A dictionary of values for each `InputAttribute`."""
        inputs_dict = {}
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, InputParameter):
                inputs_dict[name] = getattr(self, name)
        return inputs_dict

    @cached_property
    def results_dict(self) -> dict[str, Any]:
        """A combined dictionary of results & inputs."""
        return {k: self.results.get(k) for k in ["drag", "lift", "moment"]}

    @step(condition=lambda self: not (self.case_dir / "configDict").exists())
    def create_input_files(self) -> None:
        print("Creating input files")
        case_dir_base = Path.cwd() / "flat_plate_base"
        shutil.copytree(case_dir_base, self.case_dir)
        with (self.case_dir / "configDict").open("w") as fp:
            fp.write("baseDict: './configDict.base'\n")
            fp.write(f"Fr: {self.froude_num}\n")
            fp.write(f"AOA: {self.angle_of_attack}\n")

    @step(condition=lambda self: not (self.case_dir / "mesh").exists())
    def generate_mesh(self) -> None:
        print("Generating mesh")
        subprocess.run(["planingfsi", "mesh"], cwd=str(self.case_dir))

    @step(condition=lambda self: not list(self.case_dir.glob("[0-9]*")))
    def run_planingfsi(self) -> None:
        print("Running planingfsi")
        subprocess.run(["planingfsi", "run"], cwd=str(self.case_dir))
