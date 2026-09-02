"""ImpactX beam data <-> ParticleGroup.

[ImpactX](https://impactx.readthedocs.io) enables high-performance modeling of beam
dynamics in particle accelerators with collective effects. This is the next
generation of the IMPACT-Z code. ImpactX runs on modern GPUs or CPUs alike, provides
user-friendly interfaces suitable for AI/ML workflows, has many benchmarks to ensure
its correctness, and an extensive documentation.

ImpactX models particle beams with respect to a common ``s`` (in this repo called ``z``)
variable for the reference trajectory, with a spread in arrival time: all ``z`` equal,
``t`` varying, so the conversion is a direct algebraic map, like the Bmad interface and
unlike time-integrating codes (e.g., WarpX, ASTRA, etc.).

Coordinates and frames
----------------------
ImpactX describes each particle at fixed ``s`` by ``(x, y, t, px, py, pt)``:

- ``x``, ``y`` [m] are the transverse displacement from the reference particle, in the
  local (curvilinear) frame that follows the reference orbit.
- ``t`` [m] is ``c`` times the difference between the particle's and the reference
  particle's arrival time, i.e. a length, not a time.
- ``px``, ``py``, ``pt`` are dimensionless, normalized by the magnitude of the
  reference momentum: ``px = Delta(beta_x gamma) / (beta_0 gamma_0)`` and
  ``pt = -Delta(gamma) / (beta_0 gamma_0)``.

See: https://impactx.readthedocs.io/en/latest/theory/coordinates_units.html

`ParticleGroup` is a lab-frame container, so the mapping has to choose a frame:

- The transverse coordinates stay in the **local frame**: ``x`` and ``y`` are the
  displacement from the reference particle and ``z`` is zero, the reference plane.
  Adding ``x_ref``/``z_ref`` would be wrong wherever the reference orbit is bent,
  because local ``x`` is then not lab ``x``. Use `ImpactXRefPart` (``x``, ``y``, ``z``,
  ``s``, ``px``, ``py``, ``pz``) if you need to place the bunch in the lab.
- The time is **absolute**: ``t = t_ref + position_t / c``. That one is unambiguous, it
  is what openPMD's ``position/t + positionOffset/t`` means in ImpactX output, and it
  keeps quantities like `ParticleGroup.average_current` meaningful.

See also:

- ImpactX source: https://github.com/BLAST-ImpactX/impactx
- ImpactX manual: https://impactx.readthedocs.io
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..particles import ParticleGroup
from ..species import charge_of, e_charge, mass_of
from ..status import ParticleStatus
from ..units import c_light


__all__ = [
    "ImpactXRefPart",
    "PARTICLE_STATUS_LOST",
    "UnrepresentableParticleData",
    "beam_monitor_iterations",
    "impactx_to_particlegroup_data",
    "particle_id_from_idcpu",
    "particlegroup_to_impactx",
    "pmd_species_of",
    "read_beam_monitor",
    "read_beam_monitor_data",
    "refpart_from_openpmd",
]


#: ImpactX's built-in species names mapped to openPMD-beamphysics names.
#: ImpactX only knows these four; anything else needs explicit mass and charge.
IMPACTX_TO_PMD_SPECIES = {
    "electron": "electron",
    "positron": "positron",
    "proton": "proton",
    "Hminus": "H-",
}
PMD_TO_IMPACTX_SPECIES = {v: k for k, v in IMPACTX_TO_PMD_SPECIES.items()}


# --------------------------------------------------------------------------------------
# Reference particle
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactXRefPart:
    """An ImpactX reference particle, detached from any live ``ImpactX`` session.

    Holding this as a plain dataclass rather than wrapping ``impactx.RefPart`` is what
    lets the converters run with no ImpactX object in the process -- which the openPMD
    reader and the whole test suite rely on.

    Attributes
    ----------
    x, y, z : float
        Lab-frame position of the reference particle, in metres.
    t : float
        ``c * t`` of the reference particle, in **metres** (ImpactX convention).
    px, py, pz : float
        Lab-frame momenta normalized by ``m * c``, i.e. ``beta_i * gamma``.
        Dimensionless.
    pt : float
        ``-gamma`` of the reference particle. Dimensionless.
    mass_MeV : float
        Rest mass in MeV.
    charge_qe : float
        Charge in units of the elementary charge, e.g. -1 for an electron.
    s : float
        Integrated path length along the reference orbit, in metres.
    gyromagnetic_anomaly : float
        Anomalous magnetic moment, dimensionless. Carried for round-tripping only;
        `ParticleGroup` has no spin.
    """

    x: float
    y: float
    z: float
    t: float
    px: float
    py: float
    pz: float
    pt: float
    mass_MeV: float
    charge_qe: float
    s: float = 0.0
    gyromagnetic_anomaly: float = 0.0

    @property
    def mass_eV(self) -> float:
        """Rest mass in eV."""
        return self.mass_MeV * 1.0e6

    @property
    def gamma(self) -> float:
        """Relativistic gamma of the reference particle."""
        return -self.pt

    @property
    def beta_gamma(self) -> float:
        """Magnitude of the normalized reference momentum."""
        return float(np.sqrt(self.px**2 + self.py**2 + self.pz**2))

    @property
    def qm_eV(self) -> float:
        """Charge over mass in 1/eV, the form ``add_n_particles`` expects."""
        return self.charge_qe / self.mass_eV

    @property
    def qm_SI(self) -> float:
        """Charge over mass in C/kg, the form ``qm`` is written with in openPMD output."""
        return self.qm_eV * c_light**2


def _reference_is_physical(ref: ImpactXRefPart) -> bool:
    """True when ``ref`` has a positive mass and is actually moving."""
    return bool(ref.mass_MeV > 0.0 and ref.gamma >= 1.0 and ref.beta_gamma > 0.0)


def _check_reference_particle(ref: ImpactXRefPart) -> None:
    """Refuse a reference particle the conversion cannot use.

    A default-constructed ``RefPart`` has zero mass and zero energy, which makes
    ``beta_gamma`` zero and every converted momentum infinite. ImpactX writes exactly
    that into its ``particles_lost`` output, so this is not a hypothetical.

    Parameters
    ----------
    ref : ImpactXRefPart
        The reference particle to check.

    Raises
    ------
    ValueError
        If the mass is not positive or the particle is not moving.
    """
    if _reference_is_physical(ref):
        return
    raise ValueError(
        f"The reference particle is not physical: mass_MeV={ref.mass_MeV}, "
        f"gamma={ref.gamma}, beta_gamma={ref.beta_gamma}. A mass and an energy are "
        "both required to convert ImpactX' normalized coordinates."
    )


def pmd_species_of(ref: ImpactXRefPart, rtol: float = 1e-6) -> str:
    """Infer the openPMD-beamphysics species name from a reference particle.

    Parameters
    ----------
    ref : ImpactXRefPart
        The reference particle.
    rtol : float
        Relative tolerance for the mass match. ImpactX and openPMD-beamphysics carry
        electron masses that differ in the 9th digit, so this cannot be exact.

    Returns
    -------
    str
        A species name such as ``"electron"``.

    Raises
    ------
    ValueError
        If no known species matches; pass ``species=`` explicitly in that case.
    """
    for pmd_name in IMPACTX_TO_PMD_SPECIES.values():
        charge_matches = np.isclose(
            ref.charge_qe, charge_of(pmd_name) / e_charge, rtol=rtol
        )
        mass_matches = np.isclose(ref.mass_eV, mass_of(pmd_name), rtol=rtol)
        if charge_matches and mass_matches:
            return pmd_name
    raise ValueError(
        f"Cannot infer a species from charge_qe={ref.charge_qe} and "
        f"mass_MeV={ref.mass_MeV}. Pass species= explicitly."
    )


def _check_species_matches_reference(
    species: str, ref: ImpactXRefPart, rtol: float = 1e-6
) -> None:
    """Refuse a species that is not the one the reference particle describes.

    The momenta are un-normalized with the *reference* mass, and ``ParticleGroup`` then
    reads them back with the *species* mass. If the two disagree nothing raises on its
    own: the bunch is silently relabelled and comes back at the wrong energy -- a 2 GeV
    electron beam read as protons reports gamma = 2.35.

    Parameters
    ----------
    species : str
        openPMD-beamphysics species name.
    ref : ImpactXRefPart
        The reference particle the coordinates are relative to.
    rtol : float
        Relative tolerance. ImpactX and openPMD-beamphysics carry electron masses that
        differ in the 9th digit, so this cannot be exact.

    Raises
    ------
    ValueError
        If the species' rest mass or charge does not match the reference particle's.
    """
    # named for the reference particle, not just "matches": lume-impactx, which shares
    # this module verbatim, has its own _check_species_matches(pg, ref) for injection
    mass_matches = np.isclose(mass_of(species), ref.mass_eV, rtol=rtol)
    charge_matches = np.isclose(charge_of(species) / e_charge, ref.charge_qe, rtol=rtol)
    if mass_matches and charge_matches:
        return
    raise ValueError(
        f"species={species!r} (mass {mass_of(species):.6e} eV, charge "
        f"{charge_of(species) / e_charge:+.3f} e) is not the species the reference "
        f"particle describes (mass {ref.mass_eV:.6e} eV, charge {ref.charge_qe:+.3f} "
        "e). The momenta are normalized by the reference mass, so relabelling alone "
        "would return the bunch at the wrong energy. Pass species= only to name a "
        "species that pmd_species_of() cannot infer, not to convert between species."
    )


# --------------------------------------------------------------------------------------
# Coordinate conversion
# --------------------------------------------------------------------------------------


def particlegroup_to_impactx(pg: ParticleGroup, ref: ImpactXRefPart) -> dict:
    """Convert a ``ParticleGroup`` to ImpactX fixed-s beam arrays.

    Parameters
    ----------
    pg : ParticleGroup
        The bunch to convert. Unless it already sits exactly on one plane it is copied
        and drifted to its own mean ``z``, so the input is never mutated.

        ``pg.x`` and ``pg.y`` are taken to be *already relative to the reference
        particle*, which is the frame the reader returns and the frame ImpactX works
        in; ``ref.x`` and ``ref.y`` are not subtracted. A lab-frame bunch around a
        reference orbit that is off-axis must be shifted first. ``pg.z`` does not enter
        the result at all: in the local frame the plane *is* the reference particle's
        location.

        ``pg.status`` and ``pg.id`` are not carried -- ImpactX has no equivalent of the
        first, and ``add_n_particles`` assigns its own ids -- so filter dead particles
        beforehand if that matters.
    ref : ImpactXRefPart
        The reference particle the ImpactX coordinates are relative to.

    Returns
    -------
    dict
        Keys ``position_x``, ``position_y``, ``position_t``, ``momentum_x``,
        ``momentum_y``, ``momentum_t`` (arrays), ``weighting`` (array, real particles
        per macroparticle), ``qm`` (scalar, 1/eV) and ``species`` (str) -- the names
        ``ImpactXParticleContainer.to_df()`` uses, which is also what
        :func:`impactx_to_particlegroup_data` reads back.

        ``ImpactXParticleContainer.add_n_particles`` takes the same quantities under
        shorter names and has no species argument, so feed it as
        ``add_n_particles(x=data["position_x"], ..., px=data["momentum_x"], ...,
        qm=data["qm"], w=data["weighting"])``.
    """
    # The bunch must occupy a single plane. A t-coordinate bunch (spread in z) is
    # drifted to its own mean z on a copy, so the input is never mutated.
    _check_reference_particle(ref)

    if not pg.in_z_coordinates:
        pg = pg.copy()
        pg.drift_to_z()

    mass_eV = ref.mass_eV
    beta_gamma = ref.beta_gamma

    position_x = pg.x
    position_y = pg.y
    position_t = c_light * pg.t - ref.t

    momentum_x = pg.px / mass_eV / beta_gamma
    momentum_y = pg.py / mass_eV / beta_gamma

    # gamma - gamma_ref through the algebraic identity
    # (gamma^2 - gamma_ref^2) / (gamma + gamma_ref), which avoids subtracting two
    # numbers that are both ~4000 for a 2 GeV beam and differ by ~1e-3. The float64
    # representation of pg.p already sets the accuracy floor, so the gain over the
    # plain difference is a small constant factor rather than orders of magnitude --
    # but it is free.
    p_mc2 = (pg.p / mass_eV) ** 2
    gamma = np.sqrt(1.0 + p_mc2)
    dgamma = (p_mc2 - beta_gamma**2) / (gamma + ref.gamma)
    momentum_t = -dgamma / beta_gamma

    return {
        "position_x": position_x,
        "position_y": position_y,
        "position_t": position_t,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "momentum_t": momentum_t,
        "weighting": pg.weight / abs(charge_of(pg.species)),
        "qm": ref.qm_eV,
        "species": pg.species,
    }


def impactx_to_particlegroup_data(
    data: dict,
    ref: ImpactXRefPart,
    species: str | None = None,
) -> dict:
    """Convert ImpactX fixed-s beam arrays to ``ParticleGroup`` data.

    The inverse of :func:`particlegroup_to_impactx`.

    Parameters
    ----------
    data : dict
        Arrays keyed as ``ImpactXParticleContainer.to_df()`` names them:
        ``position_x/y/t``, ``momentum_x/y/t``, ``weighting``. Optional ``id`` and
        ``status`` keys are carried through.
    ref : ImpactXRefPart
        The reference particle the ImpactX coordinates are relative to.
    species : str, optional
        openPMD-beamphysics species name. Inferred from ``ref`` when omitted.

    Returns
    -------
    dict
        Suitable for ``ParticleGroup(data=...)``: ``x``, ``y``, ``z`` in metres,
        ``px``, ``py``, ``pz`` in eV/c, ``t`` in seconds, ``weight`` in Coulomb,
        ``status`` and ``species``. The result is in z-coordinates: every ``z`` is
        zero, the reference plane, and the bunch length shows up as a spread in ``t``.

    Raises
    ------
    ValueError
        If a particle's transverse momentum exceeds its total momentum, which makes
        ``pz`` imaginary. That means ``data`` and ``ref`` do not belong together.
    """
    _check_reference_particle(ref)

    if species is None:
        species = pmd_species_of(ref)
    else:
        _check_species_matches_reference(species, ref)

    mass_eV = ref.mass_eV
    beta_gamma = ref.beta_gamma
    n = len(np.asarray(data["position_x"]))

    # In the local frame the reference particle is the origin: its transverse momentum
    # is zero by construction and its longitudinal momentum is |p_ref|. ref.px / ref.pz
    # are *lab* components and must not be mixed in here.
    px_mc = beta_gamma * np.asarray(data["momentum_x"], dtype=float)
    py_mc = beta_gamma * np.asarray(data["momentum_y"], dtype=float)
    # ref.gamma is -ref.pt, so this is gamma_ref + (gamma - gamma_ref) = gamma.
    gamma = ref.gamma - beta_gamma * np.asarray(data["momentum_t"], dtype=float)

    pz_mc2 = gamma**2 - 1.0 - px_mc**2 - py_mc**2
    n_bad = int(np.count_nonzero(pz_mc2 < 0.0))
    if n_bad:
        raise ValueError(
            f"{n_bad} of {n} particles have a transverse momentum larger than their "
            "total momentum, so pz would be imaginary. The reference particle most "
            "likely does not belong to this bunch: check mass_MeV, pt and the "
            "momentum normalization."
        )
    pz_mc = np.sqrt(pz_mc2)

    if "status" in data:
        status = np.asarray(data["status"])
    else:
        status = np.full(n, int(ParticleStatus.ALIVE))

    pg_data = {
        "x": np.asarray(data["position_x"], dtype=float),
        "y": np.asarray(data["position_y"], dtype=float),
        "z": np.zeros(n),  # Zero by definition in z-coordinates
        "px": px_mc * mass_eV,
        "py": py_mc * mass_eV,
        "pz": pz_mc * mass_eV,
        "t": (ref.t + np.asarray(data["position_t"], dtype=float)) / c_light,
        "weight": np.asarray(data["weighting"], dtype=float) * abs(charge_of(species)),
        "status": status,
        "species": species,
    }
    if "id" in data:
        pg_data["id"] = np.asarray(data["id"])
    return pg_data


# --------------------------------------------------------------------------------------
# Per-particle data ParticleGroup cannot hold
# --------------------------------------------------------------------------------------

#: The spin components, which ImpactX always allocates and always writes. They stay at
#: exactly zero unless the beam was seeded with a spin distribution -- ``sim.spin =
#: True`` alone is not enough, the gate is the ``spin_distr`` argument to
#: ``add_particles``. So testing for "any non-zero" is exact: zero means there is
#: genuinely nothing to lose.
SPIN_COLUMNS = ("spin_x", "spin_y", "spin_z")


class UnrepresentableParticleData(NotImplementedError):
    """Raised when a bunch carries per-particle data ``ParticleGroup`` cannot hold.

    Converting anyway would return a bunch that looks right and has silently lost
    physics, so the conversion refuses instead. Pass ``strict=False`` to
    :func:`read_beam_monitor` to drop the extra data deliberately.
    """


def _unrepresentable_in(columns: dict) -> list[str]:
    """Name the per-particle data in ``columns`` that ``ParticleGroup`` cannot hold.

    Parameters
    ----------
    columns : dict
        Extra per-particle arrays keyed by ImpactX SoA name, i.e. everything beyond
        the coordinates, weighting and id that this module maps.

    Returns
    -------
    list of str
        Human-readable descriptions, empty when the bunch converts losslessly.
    """
    carried = []
    if any(
        name in columns and np.any(np.asarray(columns[name]) != 0.0)
        for name in SPIN_COLUMNS
    ):
        carried.append("spin (spin_x/y/z)")
    runtime = sorted(name for name in columns if name not in SPIN_COLUMNS)
    if runtime:
        carried.append(f"runtime components {runtime}")
    return carried


def _check_representable(columns: dict) -> None:
    """Refuse to convert a bunch whose extra per-particle data would be dropped.

    Parameters
    ----------
    columns : dict
        Extra per-particle arrays keyed by ImpactX SoA name.

    Raises
    ------
    UnrepresentableParticleData
        If any spin component is non-zero, or any runtime component is present.
    """
    carried = _unrepresentable_in(columns)
    if not carried:
        return
    raise UnrepresentableParticleData(
        f"This bunch carries {' and '.join(carried)}, which ParticleGroup cannot "
        "represent. Converting would silently drop it. Pass strict=False to drop it "
        "deliberately, or work with the ImpactX particle container directly. Note "
        "that ImpactX's particles_lost output always carries a runtime 's_lost' "
        "component."
    )


# --------------------------------------------------------------------------------------
# openPMD BeamMonitor reader
#
# ImpactX's elements.BeamMonitor writes standard openPMD with the species always named
# "beam" -- also in the particles_lost output, where only the *file* is named
# differently. It records position/{x,y,t}, momentum/{x,y,t}, positionOffset/{x,y,t},
# weighting, qm, spin/{x,y,z} and id, plus the reference particle and the reduced beam
# characteristics as per-iteration species attributes. Verified against ImpactX 26.08.
#
# This path needs no live ImpactX object, which is what makes ImpactXRefPart a plain
# dataclass rather than a wrapper around impactx.RefPart.
# --------------------------------------------------------------------------------------

#: openPMD records this reader consumes or deliberately ignores. Everything else in a
#: species is per-particle data ParticleGroup has no place for -- ImpactX's ``spin``,
#: or a runtime component such as the ``s/lost`` of the particles_lost output.
#:
#: ``positionOffset`` holds ``(x_ref, y_ref, t_ref)``, not zeros: the longitudinal part
#: is applied via the ``t_ref`` attribute, the transverse part is deliberately not, see
#: the module docstring. ``qm`` is redundant with the reference particle's mass and
#: charge, which are what this reader uses.
_CONSUMED_RECORDS = frozenset(
    {
        "position",
        "positionOffset",
        "momentum",
        "weighting",
        "qm",
        "id",
    }
)

#: ImpactX writes this zero-extent placeholder instead of particle records when a
#: BeamMonitor is configured with ``particles=False`` (moments-only output).
_EMPTY_PLACEHOLDER_RECORD = "empty"

#: The runtime component ImpactX attaches to every particle in its ``particles_lost``
#: output: the path length ``s`` at which the particle was lost.
_S_LOST_COLUMN = "s_lost"

#: ``status`` for a particle ImpactX has lost.
#:
#: `ParticleStatus` defines only ``CATHODE = 0`` and ``ALIVE = 1``; `ParticleGroup`
#: counts ``status == 1`` as alive and everything else as dead, and each interface
#: passes its own source code's value straight through -- `beamphysics.interfaces.bmad`
#: hands Bmad's ``state`` over as ``status`` verbatim, loss codes and all. There is no
#: universal "lost" value to reach for, so this is a choice.
#:
#: The one value it must not be is ``0``: that is a positive claim that the particle is
#: sitting at the source, and `beamphysics.interfaces.astra` writes ``status == 0`` back
#: out as Astra's ``-1``, "at the cathode". ``2`` is outside Bmad's loss-direction range
#: and carries no such meaning -- it reads as simply "not alive".
PARTICLE_STATUS_LOST = 2

# AMReX packs a particle's identity into one uint64 ``idcpu``: bit 63 marks the
# particle valid, bits 24-62 hold the id, bits 0-23 the originating MPI rank. The id
# counter is per-rank, so the id on its own repeats across the ranks of a parallel run
# -- only the packed value is globally unique, and that is what becomes the
# ``ParticleGroup`` id.
_AMREX_VALID_BIT = np.uint64(1) << np.uint64(63)


def particle_id_from_idcpu(idcpu) -> tuple[np.ndarray, np.ndarray]:
    """Split AMReX's packed ``idcpu`` into a ``ParticleGroup`` id and a validity flag.

    ImpactX writes the raw ``idcpu`` as the openPMD ``id`` record. Only the validity
    bit is stripped off here, for two reasons: with it set the value exceeds the range
    of a signed 64-bit integer, which is what ``ParticleGroup`` stores ids in, and
    aliveness belongs in ``status`` rather than in the id. Everything identifying the
    particle -- AMReX' per-rank id *and* the rank it came from -- is kept, so the
    result is unique across a parallel run where the id alone would not be.

    The original value is recovered as ``np.uint64(id) | (np.uint64(1) << 63)`` for a
    live particle; AMReX' own id and rank are ``id >> 24`` and ``id & 0xFFFFFF``.

    Parameters
    ----------
    idcpu : array_like of uint64
        The ``id`` record as stored.

    Returns
    -------
    ids : np.ndarray of int64
        ``idcpu`` with the validity bit cleared.
    valid : np.ndarray of bool
        True where the particle is marked valid.
    """
    idcpu = np.asarray(idcpu, dtype=np.uint64)
    valid = (idcpu & _AMREX_VALID_BIT).astype(bool)
    return (idcpu & ~_AMREX_VALID_BIT).astype(np.int64), valid


def refpart_from_openpmd(species: Any) -> ImpactXRefPart:
    """Rebuild a reference particle from a BeamMonitor species' attributes.

    Parameters
    ----------
    species : openpmd_api.ParticleSpecies
        A species from a BeamMonitor iteration.

    Returns
    -------
    ImpactXRefPart
        ``mass_ref`` is stored in kg and ``charge_ref`` in Coulomb, so both are
        converted here to the MeV / elementary-charge units the converters use.

    Raises
    ------
    KeyError
        If the species carries no reference particle attributes, i.e. it was not
        written by an ImpactX BeamMonitor.
    """
    attributes = set(species.attributes)
    required = {
        "x_ref", "y_ref", "z_ref", "t_ref",
        "px_ref", "py_ref", "pz_ref", "pt_ref",
        "mass_ref", "charge_ref", "s_ref",
    }  # fmt: skip
    missing = sorted(required - attributes)
    if missing:
        raise KeyError(
            f"Species is missing the reference particle attributes {missing}. "
            "This does not look like ImpactX BeamMonitor output."
        )

    get = species.get_attribute
    return ImpactXRefPart(
        x=get("x_ref"),
        y=get("y_ref"),
        z=get("z_ref"),
        t=get("t_ref"),
        px=get("px_ref"),
        py=get("py_ref"),
        pz=get("pz_ref"),
        pt=get("pt_ref"),
        mass_MeV=get("mass_ref") * c_light**2 / e_charge / 1.0e6,
        charge_qe=get("charge_ref") / e_charge,
        s=get("s_ref"),
        gyromagnetic_anomaly=(
            get("gyromagnetic_anomaly_ref")
            if "gyromagnetic_anomaly_ref" in attributes
            else 0.0
        ),
    )


def _import_openpmd_api():
    """Import ``openpmd_api``, with an actionable message when it is missing."""
    try:
        import openpmd_api
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Reading ImpactX BeamMonitor output needs the openpmd-api Python package: "
            "conda install -c conda-forge openpmd-api, or pip install openpmd-api."
        ) from exc
    return openpmd_api


def read_beam_monitor_data(
    path: str | pathlib.Path,
    iteration: int | None = None,
    species_name: str = "beam",
    species: str | None = None,
    strict: bool = True,
    ref: ImpactXRefPart | None = None,
) -> dict:
    """Read an ImpactX ``BeamMonitor`` openPMD file into ``ParticleGroup`` data.

    See :func:`read_beam_monitor` for the parameters; this is the same reader without
    the final ``ParticleGroup`` construction.

    Returns
    -------
    dict
        Suitable for ``ParticleGroup(data=...)``.
    """
    io = _import_openpmd_api()

    series = io.Series(str(path), io.Access.read_only)
    try:
        iterations = list(series.iterations)
        if not iterations:
            raise KeyError(f"No iterations in {str(path)!r}.")
        if iteration is None:
            iteration = iterations[-1]
        elif iteration not in iterations:
            raise KeyError(
                f"Iteration {iteration} not in {str(path)!r}; have {iterations}."
            )

        particles = series.iterations[iteration].particles
        if species_name not in particles:
            raise KeyError(
                f"Species {species_name!r} not in {str(path)!r}; have "
                f"{list(particles)}. ImpactX always names it 'beam', including in "
                "particles_lost output."
            )
        beam = particles[species_name]

        records = list(beam)
        if _EMPTY_PLACEHOLDER_RECORD in records:
            raise KeyError(
                f"{str(path)!r} iteration {iteration} holds no particles: this "
                "BeamMonitor was configured with particles=False and wrote only the "
                "reduced beam characteristics, which are on the species' attributes."
            )

        def _load(record_name: str, component: str):
            record_component = beam[record_name][component]
            return record_component, record_component.load_chunk()

        components = {
            "position_x": _load("position", "x"),
            "position_y": _load("position", "y"),
            "position_t": _load("position", "t"),
            "momentum_x": _load("momentum", "x"),
            "momentum_y": _load("momentum", "y"),
            "momentum_t": _load("momentum", "t"),
            "weighting": _load("weighting", io.Record_Component.SCALAR),
        }
        idcpu = beam["id"][io.Record_Component.SCALAR].load_chunk()

        # ParticleGroup cannot hold spin or runtime components, so collect them --
        # either to refuse loudly, or to report what strict=False threw away.
        extras = {}
        for name in records:
            if name in _CONSUMED_RECORDS:
                continue
            for component, record_component in beam[name].items():
                key = (
                    name
                    if component == io.Record_Component.SCALAR
                    else f"{name}_{component}"
                )
                extras[key] = record_component.load_chunk()

        file_ref = refpart_from_openpmd(beam)
        series.flush()

        # openPMD stores values that must be multiplied by unitSI to reach the unit
        # the record declares, and ImpactX writes 1.0 throughout. The positions are
        # genuinely lengths, so the factor is applied. The momenta and the weighting
        # are ImpactX' own dimensionless quantities -- normalized by the reference
        # momentum, and a count of real particles -- so a factor other than 1 would
        # mean the file no longer follows the convention decoded below. Refuse rather
        # than scale them into silent nonsense.
        data = {}
        for key, (record_component, chunk) in components.items():
            unit_si = record_component.unit_SI
            if key.startswith("position_"):
                data[key] = np.asarray(chunk) * unit_si
            elif unit_si != 1.0:
                raise ValueError(
                    f"{key} in {str(path)!r} has unitSI={unit_si}, but this reader "
                    "decodes ImpactX' dimensionless convention, in which the momenta "
                    "are normalized by the reference momentum and the weighting "
                    "counts real particles."
                )
            else:
                data[key] = np.asarray(chunk)
        extras = {key: np.asarray(chunk) for key, chunk in extras.items()}
        ids, valid = particle_id_from_idcpu(idcpu)
    finally:
        series.close()

    if ref is None:
        ref = file_ref
        if not _reference_is_physical(ref):
            raise ValueError(
                f"{str(path)!r} iteration {iteration} carries a zeroed reference "
                f"particle (mass_ref={ref.mass_MeV} MeV, gamma_ref={ref.gamma}), so "
                "its normalized coordinates cannot be converted. ImpactX wrote this "
                "into its particles_lost output before BLAST-ImpactX/impactx#1647; "
                "newer files carry a usable one. Pass ref= with the ImpactXRefPart of "
                "a BeamMonitor iteration, which refpart_from_openpmd() reads from the "
                "monitor file."
            )

    if strict:
        _check_representable(extras)
    else:
        dropped = _unrepresentable_in(extras)
        if dropped:
            warnings.warn(
                f"Dropping {' and '.join(dropped)} from {str(path)!r} iteration "
                f"{iteration}: ParticleGroup cannot represent it.",
                stacklevel=2,
            )

    data["id"] = ids
    # AMReX' validity bit is not aliveness: ImpactX copies lost particles into a
    # separate container and marks them valid *there*, so the bit is True for every
    # particle in a particles_lost file and taking it at face value would report a
    # bunch that is entirely alive.
    #
    # The s_lost runtime component identifies such a file: CollectLost always adds it.
    # The zeroed reference particle is a second signature, but only because ImpactX
    # fails to set one -- keyed on that alone, this would silently go back to reporting
    # lost particles as alive the day ImpactX fixes it.
    from_lost_file = _S_LOST_COLUMN in extras or not _reference_is_physical(file_ref)
    alive = valid & (not from_lost_file)
    data["status"] = np.where(alive, int(ParticleStatus.ALIVE), PARTICLE_STATUS_LOST)

    return impactx_to_particlegroup_data(data, ref, species=species)


def read_beam_monitor(
    path: str | pathlib.Path,
    iteration: int | None = None,
    species_name: str = "beam",
    species: str | None = None,
    strict: bool = True,
    ref: ImpactXRefPart | None = None,
) -> ParticleGroup:
    """Read an ImpactX ``BeamMonitor`` openPMD file into a ``ParticleGroup``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file ImpactX wrote, e.g. ``diags/openPMD/monitor.h5``. Any backend
        openpmd-api supports works, including ``.bp`` and a ``%T``-templated file-based
        series.
    iteration : int, optional
        Which iteration to read. The last one when omitted.
    species_name : str
        openPMD species to read. ImpactX always writes ``"beam"`` -- its lost-particle
        output differs in the *file* name (``particles_lost.*``), not the species name.
    species : str, optional
        openPMD-beamphysics species name; inferred from the reference particle when
        omitted.
    strict : bool
        When True (the default), refuse to read a bunch that carries per-particle data
        `ParticleGroup` cannot hold -- non-zero spin, or a runtime component such as
        the ``s_lost`` that ImpactX's ``particles_lost`` output always carries. Set it
        to False to drop that data and read the bunch anyway.
    ref : ImpactXRefPart, optional
        Reference particle to interpret the coordinates against. Taken from the file
        when omitted, which is what you want for a BeamMonitor. ImpactX wrote a
        *zeroed* reference particle into its ``particles_lost`` output before
        BLAST-ImpactX/impactx#1647, so reading such a file requires passing one, e.g.
        ``refpart_from_openpmd(series.iterations[n].particles["beam"])`` from the
        monitor file.

        Either way, a lost particle is only converted exactly if the reference energy
        did not change between where it was lost -- the file's own ``s_lost`` record --
        and the reference particle in hand. The momenta are normalized by ``beta_gamma``
        at the reference particle's own ``s``, so through an RF cavity or any other
        accelerating element the two differ, and so does the reference time. Which
        reference particle ImpactX stores alongside lost particles is still settling
        upstream; this reader applies no correction of its own -- it uses whatever it
        is given -- so pass ``ref=`` explicitly when it matters which one that is.

    Returns
    -------
    ParticleGroup
        In z-coordinates: every ``z`` is zero, the bunch length is a spread in ``t``,
        and the transverse coordinates are relative to the reference particle. See the
        module docstring for the frame conventions.

        ``status`` follows openPMD-beamphysics, where ``1`` is alive and anything else
        is not -- ``pg.n_alive`` and ``pg.n_dead`` split on exactly that. Every particle
        in a ``particles_lost`` file is marked :data:`PARTICLE_STATUS_LOST`, because
        they are all lost by construction.

    Raises
    ------
    KeyError
        If the requested iteration or species is not in the file, if the species is
        not ImpactX BeamMonitor output, or if the monitor recorded no particles.
    UnrepresentableParticleData
        If ``strict`` and the monitor recorded spin or runtime components.
    ValueError
        If neither ``ref`` nor the file provides a usable reference particle, or if a
        record carries a ``unitSI`` this reader cannot honour.
    ImportError
        If the ``openpmd-api`` package is not installed.

    Examples
    --------
    >>> from beamphysics.interfaces.impactx import read_beam_monitor
    >>> P = read_beam_monitor("diags/openPMD/monitor.h5")  # doctest: +SKIP
    >>> P.norm_emit_x  # doctest: +SKIP
    """
    return ParticleGroup(
        data=read_beam_monitor_data(
            path,
            iteration=iteration,
            species_name=species_name,
            species=species,
            strict=strict,
            ref=ref,
        )
    )


def beam_monitor_iterations(path: str | pathlib.Path) -> list[int]:
    """List the iterations available in an ImpactX ``BeamMonitor`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file ImpactX wrote.

    Returns
    -------
    list of int
        The iteration numbers, in file order.
    """
    io = _import_openpmd_api()

    series = io.Series(str(path), io.Access.read_only)
    try:
        return list(series.iterations)
    finally:
        series.close()
