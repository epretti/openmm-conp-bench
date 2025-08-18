#!/usr/bin/env python
# conp_benchmark: Performance and accuracy benchmarks for periodic finite field
# constant potential molecular dynamics simulations.

import argparse
import itertools
import numpy
import openmm.app
import pickle
import time

# Dimensionless constants for geometry generation.
SQRT_2 = 2.0 ** 0.5
SQRT_3 = 3.0 ** 0.5
SQRT_3_2 = SQRT_3 / 2.0
SQRT_2_3 = SQRT_2 / SQRT_3

# Length scales for geometry generation.
L_OH = 1.0 * openmm.unit.angstrom
L_HH = 2.0 * SQRT_2_3 * L_OH
L_PT = 2.7647 * openmm.unit.angstrom

# Water reference box.
WATER = openmm.app.PDBFile("water.pdb")
WATER_L_X, WATER_L_Y, WATER_L_Z = WATER.topology.getUnitCellDimensions()
WATER_L_X_NM = WATER_L_X.value_in_unit(openmm.unit.nanometer)
WATER_L_Y_NM = WATER_L_Y.value_in_unit(openmm.unit.nanometer)
WATER_L_Z_NM = WATER_L_Z.value_in_unit(openmm.unit.nanometer)

# Atomic masses.
M_H = 1.008 * openmm.unit.dalton
M_O = 15.999 * openmm.unit.dalton
M_NA = 22.99 * openmm.unit.dalton
M_CL = 35.45 * openmm.unit.dalton
M_PT = 195.08 * openmm.unit.dalton

# Atom charges.
Q_H = 0.4238 * openmm.unit.elementary_charge
Q_O = -2.0 * Q_H
Q_NA = 1.0 * openmm.unit.elementary_charge
Q_CL = -Q_NA

# Lennard-Jones parameters.
A_O = 0.37122 * openmm.unit.kilojoule_per_mole ** (1.0 / 6.0) * openmm.unit.nanometer
B_O = 0.3428 * openmm.unit.kilojoule_per_mole ** (1.0 / 12.0) * openmm.unit.nanometer
SIG_O = B_O * B_O / A_O
SIG_NA = 0.21595384927721822 * openmm.unit.nanometer
SIG_CL = 0.48304528497569194 * openmm.unit.nanometer
SIG_PT = 2.5346 * openmm.unit.angstrom
EPS_O = (A_O / B_O) ** 12 / 4.0
EPS_PT = 0.3382 * openmm.unit.ev * openmm.unit.AVOGADRO_CONSTANT_NA
EPS_NA = 1.4754532912 * openmm.unit.kilojoule_per_mole
EPS_CL = 0.05349244 * openmm.unit.kilojoule_per_mole

# Constant potential parameters.
ETA = 1.805132 / openmm.unit.angstrom
L_TF = 5.0 * openmm.unit.angstrom
V_TF = 16.21159 * openmm.unit.angstrom ** 3
VOLT = openmm.unit.volt * openmm.unit.AVOGADRO_CONSTANT_NA

# Simulation parameters.
CONC = 1.0 * openmm.unit.molar
TEMP = 300.0 * openmm.unit.kelvin
PRESS = 1.0 * openmm.unit.atmosphere
R_CUT = 1.0 * openmm.unit.nanometer
T_STEP = 2.0 * openmm.unit.femtosecond
T_LANGEVIN = 100.0 * T_STEP
N_REPORT = 100
N_EQUIL = 100000
N_PROD = 100000
F_WARM = 0.1
PLATFORM = None

def generate_fcc_lattice(n_x, n_y, n_z):
    """
    Generates orthorhombic periodic box side lengths and position coordinates
    for an n_x-by-n_y-by-n_z fcc lattice of Pt atoms with the (111) surface
    aligned along the xy-plane.  Also returns the indices of the z-layers of the
    Pt atoms.  n_y and n_z should be multiples of 2 and 3, respectively, for
    seamless periodic tiling.
    """

    positions = numpy.zeros((n_x * n_y * n_z, 3))
    layer = numpy.zeros(positions.shape[0], dtype=int)
    i = 0
    for i_x in range(n_x):
        for i_y in range(n_y):
            for i_z in range(n_z):
                j_z = i_z % 3
                positions[i] = (i_x + (i_y % 2 + (j_z == 1) + 0.5) / 2.0, SQRT_3_2 * (i_y + j_z / 3.0 + 1.0 / 6.0), SQRT_2_3 * (i_z + 0.5))
                layer[i] = i_z
                i += 1
    return (numpy.array([n_x, SQRT_3_2 * n_y, SQRT_2_3 * n_z], dtype=float) * L_PT, positions * L_PT, layer)

def generate_water_nacl_box(l_x, l_y, l_z_in, l_z_out):
    """
    Generates coordinates for water molecules and Na and Cl ions in a box of the
    specified size.  l_z_in specifies the thickness along the z-axis of the
    section of water to extract from the reference configuration, and l_z_out
    specifies the actual target thickness of the desired box.
    """

    n_nacl = int(round(CONC * l_x * l_y * l_z_in * openmm.unit.AVOGADRO_CONSTANT_NA))
    l_x_nm = l_x.value_in_unit(openmm.unit.nanometer)
    l_y_nm = l_y.value_in_unit(openmm.unit.nanometer)
    l_z_in_nm = l_z_in.value_in_unit(openmm.unit.nanometer)

    xyz_water = []
    for residue in WATER.topology.residues():
        atom_o, atom_h1, atom_h2 = residue.atoms()
        xyz_o_in = WATER.positions[atom_o.index]
        xyz_o = openmm.Vec3(xyz_o_in[0] / WATER_L_X % 1.0 * WATER_L_X_NM, xyz_o_in[1] / WATER_L_Y % 1.0 * WATER_L_Y_NM, xyz_o_in[2] / WATER_L_Z % 1.0 * WATER_L_Z_NM) * openmm.unit.nanometer
        xyz_o_shift = xyz_o - xyz_o_in
        xyz_h1 = WATER.positions[atom_h1.index] + xyz_o_shift
        xyz_h2 = WATER.positions[atom_h2.index] + xyz_o_shift

        for i_x in range(int(numpy.ceil(l_x / WATER_L_X))):
            for i_y in range(int(numpy.ceil(l_y / WATER_L_Y))):
                for i_z in range(int(numpy.ceil(l_z_in / WATER_L_Z))):
                    offset = openmm.Vec3(i_x * WATER_L_X_NM, i_y * WATER_L_Y_NM, i_z * WATER_L_Z_NM) * openmm.unit.nanometer
                    xyz_o_offset = xyz_o + offset
                    xyz_o_offset_nm = xyz_o_offset.value_in_unit(openmm.unit.nanometer)
                    if xyz_o_offset_nm.x < l_x_nm and xyz_o_offset_nm.y < l_y_nm and xyz_o_offset_nm.z < l_z_in_nm:
                        xyz_h1_offset_nm = (xyz_h1 + offset).value_in_unit(openmm.unit.nanometer)
                        xyz_h2_offset_nm = (xyz_h2 + offset).value_in_unit(openmm.unit.nanometer)
                        xyz_o_out_nm = openmm.Vec3(xyz_o_offset_nm[0], xyz_o_offset_nm[1], xyz_o_offset_nm[2] * (l_z_out / l_z_in))
                        xyz_water.append((xyz_o_out_nm, xyz_h1_offset_nm + (xyz_o_out_nm - xyz_o_offset_nm), xyz_h2_offset_nm + (xyz_o_out_nm - xyz_o_offset_nm)))

    xyz_na = []
    xyz_cl = []
    for i_nacl, i_water in enumerate(sorted(numpy.random.choice(numpy.arange(len(xyz_water)), 2 * n_nacl, False))[::-1]):
        (xyz_na, xyz_cl)[i_nacl % 2].append(xyz_water[i_water][0])
        del xyz_water[i_water]

    return numpy.array(xyz_water).reshape(-1, 3, 3) * openmm.unit.nanometer, numpy.array(xyz_na).reshape(-1, 3) * openmm.unit.nanometer, numpy.array(xyz_cl).reshape(-1, 3) * openmm.unit.nanometer

def create_electrode_system(n_electrode, n_electrolyte, aspect_ratio, double_cell, l_z=None):
    """
    Generates a system with the given number of electrode particles and
    electrolyte particles and a desired aspect ratio.  If the double_cell flag
    is set, the system will be configured for a double-cell rather than a
    finite-field constant potential simulation.  If l_z is specified, this
    should be a desired box thickness determined by an equilibration run.
    """

    v_electrode_target = n_electrode * L_PT * L_PT * L_PT / SQRT_2
    v_electrolyte_target = n_electrolyte / (WATER.topology.getNumAtoms() / (WATER_L_X * WATER_L_Y * WATER_L_Z) - 4.0 * CONC * openmm.unit.AVOGADRO_CONSTANT_NA)
    f_target = v_electrode_target / (2.0 * (v_electrode_target + v_electrolyte_target))
    m_z = max(int(round((n_electrode * f_target * f_target * aspect_ratio * aspect_ratio / (24 * SQRT_3)) ** (1.0 / 3.0))), 1)
    m_y = max(int(round((n_electrode / (12 * SQRT_3 * m_z)) ** 0.5)), 1)
    m_x = max(int(round(n_electrode / (12 * m_y * m_z))), 1)
    n_z = 3 * m_z
    (l_x, l_y, l_z_electrode), xyz_pt, layer = generate_fcc_lattice(m_x, 2 * m_y, n_z)

    if double_cell:
        # Electrode, electrolyte, electrode, electrolyte
        l_z_electrolyte_in = v_electrolyte_target / (2.0 * l_x * l_y)
        l_z_electrolyte_out = l_z_electrolyte_in if l_z is None else l_z / 2.0 - l_z_electrode
        xyz_water, xyz_na, xyz_cl = generate_water_nacl_box(l_x, l_y, l_z_electrolyte_in, l_z_electrolyte_out)
        shift_1 = (0.0, 0.0, l_z_electrode.value_in_unit(openmm.unit.nanometer))
        shift_2 = (0.0, 0.0, (l_z_electrode + l_z_electrolyte_out).value_in_unit(openmm.unit.nanometer))

        xyz_pt = numpy.concatenate((xyz_pt.value_in_unit(openmm.unit.nanometer), xyz_pt.value_in_unit(openmm.unit.nanometer) + shift_2)) * openmm.unit.nanometer
        xyz_water = numpy.concatenate((xyz_water.value_in_unit(openmm.unit.nanometer) + shift_1, xyz_water.value_in_unit(openmm.unit.nanometer) + shift_1 + shift_2)) * openmm.unit.nanometer
        xyz_na = numpy.concatenate((xyz_na.value_in_unit(openmm.unit.nanometer) + shift_1, xyz_na.value_in_unit(openmm.unit.nanometer) + shift_1 + shift_2)) * openmm.unit.nanometer
        xyz_cl = numpy.concatenate((xyz_cl.value_in_unit(openmm.unit.nanometer) + shift_1, xyz_cl.value_in_unit(openmm.unit.nanometer) + shift_1 + shift_2)) * openmm.unit.nanometer
        electrode = numpy.concatenate((numpy.zeros(layer.size, dtype=int), numpy.ones(layer.size, dtype=int)))
        layer = numpy.concatenate((layer, layer))
        return l_x, l_y, 2.0 * (l_z_electrode + l_z_electrolyte_out), xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer

    else:
        # Electrode, electrolyte, electrode
        l_z_electrolyte_in = v_electrolyte_target / (l_x * l_y)
        l_z_electrolyte_out = l_z_electrolyte_in if l_z is None else l_z - 2.0 * l_z_electrode
        xyz_water, xyz_na, xyz_cl = generate_water_nacl_box(l_x, l_y, l_z_electrolyte_in, l_z_electrolyte_out)
        shift_1 = (0.0, 0.0, l_z_electrode.value_in_unit(openmm.unit.nanometer))
        shift_2 = (0.0, 0.0, (l_z_electrode + l_z_electrolyte_out).value_in_unit(openmm.unit.nanometer))

        xyz_pt = numpy.concatenate((xyz_pt.value_in_unit(openmm.unit.nanometer), xyz_pt.value_in_unit(openmm.unit.nanometer) + shift_2)) * openmm.unit.nanometer
        xyz_water = (xyz_water.value_in_unit(openmm.unit.nanometer) + shift_1) * openmm.unit.nanometer
        xyz_na = (xyz_na.value_in_unit(openmm.unit.nanometer) + shift_1) * openmm.unit.nanometer
        xyz_cl = (xyz_cl.value_in_unit(openmm.unit.nanometer) + shift_1) * openmm.unit.nanometer
        electrode = numpy.concatenate((numpy.zeros(layer.size, dtype=int), numpy.ones(layer.size, dtype=int)))
        layer = numpy.concatenate((layer, layer))
        return l_x, l_y, 2.0 * l_z_electrode + l_z_electrolyte_out, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer

def generate_openmm_system(l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, freeze_all, use_conp, use_matrix=None):
    """
    Generates an OpenMM system and topology for a system with the specified box
    lengths, Pt, water, Na, and Cl coordinates.  If the freeze_all flag is set,
    all Pt atoms will be frozen; otherwise, only Pt atoms in the central layer
    or layers of the electrode or electrodes will be (or, if not using constant
    potential mode, only a single Pt atom will be frozen).
    """

    n_pt = xyz_pt.shape[0]
    n_water = xyz_water.shape[0]
    n_na = xyz_na.shape[0]
    n_cl = xyz_cl.shape[0]
    print(f"Generating system with {2 * n_water} H, {n_water} O, {n_na} Na, {n_cl} Cl, and {n_pt} Pt")

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors((l_x, 0.0, 0.0), (0.0, l_y, 0.0), (0.0, 0.0, l_z))

    topology = openmm.app.Topology()
    topology.setUnitCellDimensions(openmm.unit.Quantity((l_x, l_y, l_z)))
    topology_pt = topology.addChain()
    topology_water = topology.addChain()
    topology_nacl = topology.addChain()

    nonb = openmm.NonbondedForce()
    if use_conp:
        conp = openmm.ConstantPotentialForce()

    electrode_sets = {}
    for i_pt in range(n_pt):
        system.addParticle(0.0 if freeze_all or (abs(2 * layer[i_pt] - (n_z - 1)) <= 1 if use_conp else not i_pt) else M_PT)
        electrode_sets.setdefault(electrode[i_pt], set()).add(i_pt)

        residue = topology.addResidue("PT", topology_pt)
        topology.addAtom("PT", openmm.app.element.platinum, residue)

        nonb.addParticle(0.0, SIG_PT, EPS_PT)
        if use_conp:
            conp.addParticle(0.0)

    for i_water in range(n_water):
        i_o = system.addParticle(M_O)
        i_h1 = system.addParticle(M_H)
        i_h2 = system.addParticle(M_H)
        system.addConstraint(i_o, i_h1, L_OH)
        system.addConstraint(i_o, i_h2, L_OH)
        system.addConstraint(i_h1, i_h2, L_HH)

        residue = topology.addResidue("WAT", topology_water)
        topology.addAtom("O", openmm.app.element.oxygen, residue)
        topology.addAtom("H1", openmm.app.element.hydrogen, residue)
        topology.addAtom("H2", openmm.app.element.hydrogen, residue)

        nonb.addParticle(0.0 if use_conp else Q_O, SIG_O, EPS_O)
        nonb.addParticle(0.0 if use_conp else Q_H, 1.0, 0.0)
        nonb.addParticle(0.0 if use_conp else Q_H, 1.0, 0.0)
        nonb.addException(i_o, i_h1, 0.0, 1.0, 0.0)
        nonb.addException(i_o, i_h2, 0.0, 1.0, 0.0)
        nonb.addException(i_h1, i_h2, 0.0, 1.0, 0.0)
        if use_conp:
            conp.addParticle(Q_O)
            conp.addParticle(Q_H)
            conp.addParticle(Q_H)
            conp.addException(i_o, i_h1, 0.0)
            conp.addException(i_o, i_h2, 0.0)
            conp.addException(i_h1, i_h2, 0.0)
    
    for i_na in range(n_na):
        system.addParticle(M_NA)

        residue = topology.addResidue("NA", topology_nacl)
        topology.addAtom("NA", openmm.app.element.sodium, residue)

        nonb.addParticle(0.0 if use_conp else Q_NA, SIG_PT, EPS_PT)
        if use_conp:
            conp.addParticle(Q_NA)
    
    for i_cl in range(n_cl):
        system.addParticle(M_CL)

        residue = topology.addResidue("CL", topology_nacl)
        topology.addAtom("CL", openmm.app.element.chlorine, residue)

        nonb.addParticle(0.0 if use_conp else Q_CL, SIG_CL, EPS_CL)
        if use_conp:
            conp.addParticle(Q_CL)
    
    nonb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic if use_conp else openmm.NonbondedForce.PME)
    nonb.setCutoffDistance(R_CUT)
    system.addForce(nonb)

    if use_conp:
        if use_matrix is None:
            method = openmm.ConstantPotentialForce.Matrix if freeze_all else openmm.ConstantPotentialForce.CG
        elif use_matrix is True:
            method = openmm.ConstantPotentialForce.Matrix
        elif use_matrix is False:
            method = openmm.ConstantPotentialForce.CG
        conp.setConstantPotentialMethod(use_matrix)
        conp.setUseChargeConstraint(True)
        conp.setCutoffDistance(R_CUT)
        system.addForce(conp)
        for i_electrode in sorted(electrode_sets):
            conp.addElectrode(electrode_sets[i_electrode], 0.0, 1.0 / ETA, L_TF * L_TF / V_TF)

    return system, topology, numpy.concatenate((
        xyz_pt.value_in_unit(openmm.unit.nanometer),
        xyz_water.value_in_unit(openmm.unit.nanometer).reshape((-1, 3)),
        xyz_na.value_in_unit(openmm.unit.nanometer),
        xyz_cl.value_in_unit(openmm.unit.nanometer)
    )) * openmm.unit.nanometer

def integrate_openmm_system(system, topology, positions, n_equil, n_prod, t_bench, prefix, barostat_kind, do_minimize):
    """
    Integrates an OpenMM system starting from the specified positions for a
    given number of equilibration and production steps, and writes output files
    with the specified prefix.  Set barostat_kind = 0 for an NVT simulation, 1
    for an NPT simulation with fixed box length ratios, and 2 for an NPT
    simulation with box lengths that can fluctuate independently.  The
    do_minimize flag controls whether local energy minimization is performed
    before starting.
    """

    if barostat_kind == 0:
        pass
    elif barostat_kind == 1:
        system.addForce(openmm.MonteCarloBarostat(PRESS, TEMP))
    elif barostat_kind == 2:
        system.addForce(openmm.MonteCarloAnisotropicBarostat((PRESS, PRESS, PRESS), TEMP))
    else:
        raise ValueError
    integrator = openmm.LangevinMiddleIntegrator(TEMP, 1.0 / T_LANGEVIN, T_STEP)
    if PLATFORM is None:
        context = openmm.Context(system, integrator)
    else:
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(PLATFORM))
    context.setPositions(positions)

    openmm.app.PDBFile.writeFile(topology, positions, f"{prefix}.pdb")

    if do_minimize:
        print(f"Potential energy before minimization: {context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole):24.16e}")
        openmm.LocalEnergyMinimizer.minimize(context)
        print(f"Potential energy after minimization:  {context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole):24.16e}")
        openmm.app.PDBFile.writeFile(topology, context.getState(positions=True).getPositions(), f"{prefix}.min.pdb")
    context.setVelocitiesToTemperature(TEMP)

    conp = None
    for force in system.getForces():
        if isinstance(force, openmm.ConstantPotentialForce):
            if conp is None:
                conp = force
            else:
                raise ValueError
    groups = []
    if conp is not None:
        for i_electrode in range(conp.getNumElectrodes()):
            groups.append(sorted(conp.getElectrodeParameters(i_electrode)[0]))

    for i_equil in range(0, n_equil, N_REPORT):
        integrator.step(min(N_REPORT, n_equil - i_equil))

    with open(f"{prefix}.log", "w") as log_file, open(f"{prefix}.dcd", "wb") as dcd_file:
        print("# Step KE PE Lx Ly Lz", *(f"q{i}" for i in range(len(groups))), file=log_file)
        dcd_writer = openmm.app.DCDFile(dcd_file, topology, T_STEP, 0, N_REPORT)

        def make_report(i_prod, n_step):
            state = context.getState(positions=True, energy=True, enforcePeriodicBox=True)
            vectors = state.getPeriodicBoxVectors()
            group_charges = []
            if conp is not None:
                charges = conp.getCharges(context).value_in_unit(openmm.unit.elementary_charge)
                for group in groups:
                    group_charges.append(sum(charges[i] for i in group))
            print(f"{i_prod + n_step:10}"
                f" {state.getKineticEnergy().value_in_unit(openmm.unit.kilojoule_per_mole):24.16e}"
                f" {state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole):24.16e}"
                f" {vectors[0][0].value_in_unit(openmm.unit.nanometer):24.16e}",
                f" {vectors[1][1].value_in_unit(openmm.unit.nanometer):24.16e}",
                f" {vectors[2][2].value_in_unit(openmm.unit.nanometer):24.16e}",
                *(f" {group_charge:24.16e}" for group_charge in group_charges),
                file=log_file)
            dcd_writer.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

        if n_prod is not None and t_bench is None:
            for i_prod in range(0, n_prod, N_REPORT):
                n_step = min(N_REPORT, n_prod - i_prod)
                integrator.step(n_step)
                make_report(i_prod, n_step)
        elif n_prod is None and t_bench is not None:
            # Warm up.
            t_start = time.perf_counter_ns()
            for i_prod in itertools.count(step=N_REPORT):
                integrator.step(N_REPORT)
                make_report(i_prod, N_REPORT)
                if (time.perf_counter_ns() - t_start) / 10 ** 9 > t_bench * F_WARM:
                    break

            # Benchmark.
            t_start = time.perf_counter_ns()
            bench_steps = 0
            for i_prod in itertools.count(step=N_REPORT):
                integrator.step(N_REPORT)
                bench_steps += N_REPORT
                make_report(i_prod, N_REPORT)
                t_elapsed = (time.perf_counter_ns() - t_start) / 10 ** 9
                if t_elapsed > t_bench:
                    break

            print(f"Benchmarking results for {prefix}:")
            print(f"    Simulated steps:          {bench_steps}")
            print(f"    Elapsed time (s-wall):    {t_elapsed}")
            print(f"    Cost (ns-wall/step-sim):  {round(t_elapsed * 10 ** 9 / bench_steps)}")
            print(f"    Speed (ns-sim/day-wall):  {(bench_steps * T_STEP).value_in_unit(openmm.unit.nanosecond) / (t_elapsed / 86400):.3f}")

        else:
            raise RuntimeError

    return context.getState(positions=True).getPositions()

def prepare_conp_system(n_electrode, n_electrolyte, aspect_ratio, double_cell, freeze_all):
    prefix = f"system_{n_electrode}_{n_electrolyte}_{aspect_ratio}_{int(double_cell)}_{int(freeze_all)}"

    l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer = create_electrode_system(n_electrode, n_electrolyte, aspect_ratio, double_cell)
    system, topology, positions = generate_openmm_system(l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, False, False)
    integrate_openmm_system(system, topology, positions, N_EQUIL, N_PROD, None, f"{prefix}_equilibrate_npt", 2, True)

    l_x_log, l_y_log, l_z_log = numpy.loadtxt(f"{prefix}_equilibrate_npt.log")[:, 3:].T
    l_z_target = numpy.mean(l_x_log * l_y_log * l_z_log) * openmm.unit.nanometer ** 3 / (l_x * l_y)

    l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer = create_electrode_system(n_electrode, n_electrolyte, aspect_ratio, double_cell, l_z_target)
    system, topology, positions = generate_openmm_system(l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, freeze_all, False)
    positions = integrate_openmm_system(system, topology, positions, N_EQUIL, N_PROD, None, f"{prefix}_equilibrate_nvt", 0, True)

    with open(f"{prefix}_state.dat", "wb") as file:
        pickle.dump((l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, positions), file)

def simulate_conp_system(n_electrode, n_electrolyte, aspect_ratio, double_cell, freeze_all, use_matrix, potential_v, n_prod, t_bench):
    prefix = f"system_{n_electrode}_{n_electrolyte}_{aspect_ratio}_{int(double_cell)}"

    with open(f"{prefix}_1_state.dat", "rb") as file:
        l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, positions = pickle.load(file)

    system, topology, _ = generate_openmm_system(l_x, l_y, l_z, xyz_pt, xyz_water, xyz_na, xyz_cl, n_z, electrode, layer, freeze_all, True, use_matrix)
    conp, = (force for force in system.getForces() if isinstance(force, openmm.ConstantPotentialForce))

    particles, _, width, thomas_fermi = conp.getElectrodeParameters(1)
    conp.setElectrodeParameters(1, particles, potential_v * VOLT, width, thomas_fermi)
    if not double_cell:
        conp.setExternalField((0.0, 0.0, -potential_v * VOLT / l_z))

    integrate_openmm_system(system, topology, positions, 0, n_prod, t_bench, f"{prefix}_{int(freeze_all)}_{int(use_matrix)}_run_{potential_v}", 0, False)

def prepare_conp_systems():
    # n_electrode, n_electrolyte, aspect_ratio, double_cell, freeze_all
    prepare_conp_system(3000, 6000, 3.0, False, True)
    prepare_conp_system(3000, 6000, 3.0, True, True)
    prepare_conp_system(3000, 6000, 1.5, False, True)
    prepare_conp_system(3000, 6000, 6.0, False, True)
    prepare_conp_system(1000, 8000, 3.0, False, True)
    prepare_conp_system(5000, 4000, 3.0, False, True)
    prepare_conp_system(1000, 2000, 3.0, False, True)
    prepare_conp_system(15000, 30000, 3.0, False, True)

def main():
    global PLATFORM

    benchmarks = {
        "base":        lambda t_bench, use_matrix: simulate_conp_system( 3000,  6000, 3.0, False, True,  use_matrix, 10.0, None, t_bench),
        "unfrozen":    lambda t_bench            : simulate_conp_system( 3000,  6000, 3.0, False, False, False,      10.0, None, t_bench),
        "zero":        lambda t_bench, use_matrix: simulate_conp_system( 3000,  6000, 3.0, False, True,  use_matrix, 0.0,  None, t_bench),
        "double":      lambda t_bench, use_matrix: simulate_conp_system( 3000,  6000, 3.0, True,  True,  use_matrix, 10.0, None, t_bench),
        "short":       lambda t_bench, use_matrix: simulate_conp_system( 3000,  6000, 1.5, False, True,  use_matrix, 10.0, None, t_bench),
        "long":        lambda t_bench, use_matrix: simulate_conp_system( 3000,  6000, 6.0, False, True,  use_matrix, 10.0, None, t_bench),
        "electrolyte": lambda t_bench, use_matrix: simulate_conp_system( 1000,  8000, 3.0, False, True,  use_matrix, 10.0, None, t_bench),
        "electrode":   lambda t_bench, use_matrix: simulate_conp_system( 5000,  4000, 3.0, False, True,  use_matrix, 10.0, None, t_bench),
        "small":       lambda t_bench, use_matrix: simulate_conp_system( 1000,  2000, 3.0, False, True,  use_matrix, 10.0, None, t_bench),
        "large":       lambda t_bench, use_matrix: simulate_conp_system(15000, 30000, 3.0, False, True,  use_matrix, 10.0, None, t_bench),
    }

    parser = argparse.ArgumentParser()
    for benchmark_name in benchmarks:
        parser.add_argument(f"--{benchmark_name}", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--time", type=float, default=60.0)
    parser.add_argument("--method", choices=["matrix", "cg"], required=True)
    parser.add_argument("--platform")
    args = parser.parse_args()

    PLATFORM = args.platform
    
    t_bench = args.time
    use_matrix = args.method == "matrix"

    if args.all:
        for benchmark_name, benchmark_function in benchmarks.items():
            print(benchmark_name)
            supports_use_matrix = benchmark_function.__code__.co_argcount > 1
            if use_matrix and not supports_use_matrix:
                continue
            if supports_use_matrix:
                benchmark_function(t_bench, use_matrix)
            else:
                benchmark_function(t_bench)
    else:
        for benchmark_name, benchmark_function in benchmarks.items():
            if getattr(args, benchmark_name):
                print(benchmark_name)
                supports_use_matrix = benchmark_function.__code__.co_argcount > 1
                if use_matrix and not supports_use_matrix:
                    raise ValueError(benchmark_name)
                if supports_use_matrix:
                    benchmark_function(t_bench, use_matrix)
                else:
                    benchmark_function(t_bench)


if __name__ == "__main__":
    main()
