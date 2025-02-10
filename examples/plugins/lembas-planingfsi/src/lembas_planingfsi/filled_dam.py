"""An example using `lembas` to analyze a hydrostatic dam filled with water.

Each case is run with `planingfsi` and is characterized by the reference pressure
head at the base. The dam will deform into an equilibrium state using the
finite-element solver.

"""

import dataclasses
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import NamedTuple

import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from planingfsi import FlexibleMembraneSubstructure
from planingfsi import Mesh
from planingfsi import Simulation

from lembas import Case
from lembas import CaseList
from lembas import InputParameter
from lembas import result
from lembas import step

BASE_DIR = Path(__file__).parent


@dataclasses.dataclass
class Coordinate:
    x: float
    y: float


class HydrostaticDamResults(NamedTuple):
    max_height: float
    coords: list[Coordinate]


class HydrostaticDamCase(Case):
    reference_head = InputParameter(type=float)

    @property
    def case_dir(self) -> Path:
        return BASE_DIR / "cases" / f"href={self.reference_head:0.3f}m"

    @result("max_height", "coords")
    def load_results(self) -> HydrostaticDamResults:
        """Load results from files and return."""
        # Find the latest time directory to load results from
        results_dirs = sorted(
            self.case_dir.glob("[0-9]*"), key=lambda d: int(d.name), reverse=True
        )
        results_dir = results_dirs[0]
        coords = np.loadtxt(results_dir / "coords_dam.txt")
        return HydrostaticDamResults(
            max_height=np.max(coords[:, 1]),
            coords=[Coordinate(*c) for c in coords],
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
        return {k: self.results.get(k) for k in ["max_height", "coords"]}

    @staticmethod
    def generate_mesh() -> Mesh:
        mesh = Mesh()

        # Create points (ID, type, params)
        mesh.add_point(1, "rel", [0, 0, 1.0])
        mesh.add_point(2, "rel", [0, 180, 1.0])

        dam = mesh.add_submesh("dam")
        dam.add_curve(1, 2, Nel=50, arcLen=np.pi)

        return mesh

    @step(condition=lambda case: not case.case_dir.exists())
    def run(self) -> None:
        mesh = self.generate_mesh()

        simulation = Simulation()
        simulation.case_dir = self.case_dir

        # Set some global configuration values
        simulation.config.flow.froude_num = 1.0
        simulation.config.flow.waterline_height = self.reference_head
        simulation.config.plotting._pressure_scale_pct = 1e-8
        simulation.config.solver.max_it = 500

        body = simulation.add_rigid_body()
        body.add_substructure(
            FlexibleMembraneSubstructure(
                name="dam",
                seal_pressure_method="hydrostatic",
            )
        )

        simulation.load_mesh(mesh)
        simulation.run()


def plot_summary_results(cases: CaseList[HydrostaticDamCase]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df = pandas.DataFrame.from_records(
        case.inputs_dict | case.results_dict for case in cases
    ).drop(columns="coords")
    df.plot(x="reference_head", y="max_height", ax=ax[0], label="Max Dam Height [m]")
    ax[0].set_xlabel("Reference Head [m]")

    lines, colors = [], []
    for i, case in enumerate(cases):
        coords_df = pandas.DataFrame.from_records(case.results_dict["coords"])
        lines.append(coords_df[["x", "y"]].to_numpy())
        colors.append(case.reference_head)

    lc = LineCollection(lines)
    lc.set_array(colors)

    ax[1].add_collection(lc)
    ax[1].axis("equal")

    fig.colorbar(lc).set_label("Reference Head [m]")
    plt.show()


def main() -> None:
    cases: CaseList[HydrostaticDamCase] = CaseList()
    cases.add_cases_by_parameter_sweep(
        HydrostaticDamCase, reference_head=np.arange(0.8, 10.1, 0.2)
    )
    cases.run_all()

    plot_summary_results(cases)


if __name__ == "__main__":
    main()
