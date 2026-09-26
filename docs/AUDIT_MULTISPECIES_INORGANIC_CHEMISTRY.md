# Audit: one multispecies, multiphase inorganic chemistry state

Date: 2026-09-20

Scope: the chamber atmosphere, its voxel gas and condensate, wall films and
deposits, pools, droplets, aerosols, ports, records, renderer publications,
and the managed-dt boundary.  This is an audit and implementation contract.
It does not claim that the combined law has been compiled or executed.

## Verdict

The chamber is not presently a multispecies phase-transition system.  It is a
water phase-transition law repeated through a species-shaped Python mapping,
surrounded by water-specific surface, pool, aerosol, port, recording, and
rendering code.  The mapping is useful, but it is not a chemical mixture
state.

The correct replacement is one chemical inventory carried by every material
owner.  Its conserved quantity is moles of explicit chemical species in
explicit phases:

```text
n[owner, chemical_species, phase]  mol
```

Every transport, phase transition, ionic association or dissociation,
precipitation, surface reaction, and homogeneous reaction is a stoichiometric
extent applied to that inventory.  Temperature and pressure are properties of
an owner and its mixture.  Molarity, molality, ionic strength, activities,
fugacities, saturation indices, heat capacities, reaction heat, and latent
heat are derived from the same inventory.  No subsystem may keep a second
water mass, salt count, contaminant dictionary, or anonymous dry-air mass.

The complete chemistry must be assembled as one coupled residual system and
advanced as one managed-dt participant.  Running seven laws sequentially and
passing last-step outputs between them cannot provide simultaneous chemical
or energy equilibrium.

The reduced numerical law uses the repository's `AbstractTensor` precision
surface at two limbs.  The eager authoring path promotes operands with
`Precision.of(value, 2)`.  The compiled path uses `Precision[2]` regions and
interleaved two-limb span storage, the same convention as the compiled signal
kernels.  A state publication that remains a precision state publishes both
limbs; a deliberately narrow diagnostic collapses only at that named
boundary.  SymPy source coefficients remain exact rationals until numerical
materialization; there is no separate Python-float or SymPy precision scheme.

## What exists now

### Voxel state

`ChamberSim.air` owns `m_a` and `T`.  `m_a` is anonymous dry air.  It has no
nitrogen, oxygen, argon, carbon-dioxide, or trace-species composition.

Each entry of `ChamberSim.species` owns four mass columns:

```text
m_v  vapour
m_l  suspended cloud liquid
m_i  suspended ice
m_r  rain
```

`voxel_species_step` is parameterized by one condensable's saturation curve,
latent heats, heat capacities, diffusivity, and settling speeds.  The package
sums every instantiated species' vapour partial pressure, heat capacity, and
latent heat into the shared air law.  This is the one existing path that is
already structurally multispecies.

The live native construction instantiates only `water`.  Nitrogen, oxygen,
argon, and carbon dioxide are hidden in `m_a`, so none can condense, freeze,
deposit, enrich, or leave selectively.  This is especially wrong at the
77 K cold head: nitrogen, oxygen, argon, carbon dioxide, and water have
different accessible phases and must not share an anonymous carrier mass.

### Surface state

Each surface owns:

```text
m_film       one assumed liquid, water
n_s_film     total moles of one assumed salt
m_frost      one assumed solid, water ice
m_crust      mass of that same salt
m_sed        undifferentiated sediment
T_s
```

The surface receives only the selected water species' ambient vapour pressure.
Its activity model is one salt in water.  It cannot represent mixed solvents,
multiple ions, charge balance, multiple precipitates, co-deposition of CO2,
liquid oxygen or nitrogen, adsorption, or a deposit whose composition changes
with temperature.

### Pool state

Each pool owns `m_p`, `n_s_pool`, and `T_l`: one liquid mass and one salt mole
count.  Evaporation is hard-wired to water properties and one water vapour
pressure.  Inflow carries one scalar `s_in`, and the current package never
actually supplies pool solute from surface runoff.  A pool therefore loses
species identity at the moment material enters it.

### Droplet, solution, and aerosol state

`droplet_step` represents a water droplet containing one inorganic salt.  Its
state is radius, one salt mole count, particle temperature, height, and fall
speed.  Water mass is inferred from radius.  There is no composition vector.

`salt_solution_step` represents water plus one salt split into dissolved and
crystalline amounts.  It computes molality, but it has no explicit ions,
electroneutrality, ionic strength, competing salts, complexes, acid/base
speciation, or shared precipitates.

`aerosol_step` carries only total dry number and total dry mass.  It can seed
water droplets, but it cannot say whether a particle is sodium chloride,
sulfate, soot, dust, or a mixture.  The live chamber does not instantiate the
aerosol, droplet, or salt-solution laws at all.

### Ports and engine-toy state

The live open port uses engine-toy's `Stream` and `Contaminant`, then manually
splits total incoming humid gas into dry-air mass and water-vapour mass.
`Contaminant` is a fixed list of mass ratios per kilogram of dry air.  Its
entries mix chemical species (`so2`, `co`) with physical categories
(`liquid`, `dust`, `oil`).  It has no phase axis, mole inventory, elemental
composition, charge, or reaction thermodynamics.

Engine-toy's `ThermalVolume.species_kg`, `DewarAtmosphere.kg`, air-separation
tables, fluid registry, and phase table all preserve useful facts, but each is
a separate species vocabulary.  The phase table is explicitly a behavioral
lookup, not an equilibrium solver, and most transition temperatures are
one-atmosphere points.  These records must become views or source annotations
on the chemical compendium; they cannot remain independent inventories.

### Execution and publication

`ChamberSim._advance_all` advances surfaces, pools, air, species, then aerosol.
Several couplings intentionally use the previous tick's outputs.  That order
creates operator splitting and prevents a single equilibrium or energy solve.

`LawEngine` recognizes recurrence only through `*_next` output names.  It is
an explicit state transition.  The symbolic equation compiler also requires
simultaneous outputs whose right-hand sides do not consume another output from
the same call.  A coupled equilibrium calculation therefore needs its Newton
or other nonlinear reduction authored inside the step; it cannot be expressed
as a chain of symbolic outputs and called simultaneous.

Publications are keyed by bare output names and semantics such as `LWC` and
`cloud_water_content`.  A second condensable overwrites those identities in
the recorder and live viewer.  Species and phase identity must be part of the
publication key and span metadata.

## Canonical identity and compendium

### Stable identities

The global compendium needs four independent registries:

1. **Elements and conserved components.** Atomic number, symbol, standard
   atomic weight, and optional isotope identity.
2. **Chemical species.** Formula, elemental composition vector, charge,
   molar mass, oxidation states where relevant, and allowed phases.
3. **Phases.** At minimum gas, aqueous, liquid, solid, adsorbed, and plasma.
   A chemical state is the stable pair `(species_id, phase_id)`.
4. **Reactions.** A stoichiometric row over chemical states, equilibrium or
   kinetic data, temperature/pressure validity, reaction enthalpy data, and
   provenance.

`water` is a chemical species.  `H2O(g)`, `H2O(aq or liquid)`, and `H2O(s)`
are chemical states.  A phase transition is a reaction row between those
states.  `Na+`, `Cl-`, `NaCl(aq)`, and `NaCl(s)` are separate species/states;
ionic bonding is represented by allowed, balanced species and reaction rows,
not by pairing unlike charges heuristically.

Every reaction row must pass these construction-time identities:

```text
A @ nu[r] = 0       element and isotope conservation
z @ nu[r] = 0       charge conservation
```

where `A[element, species]` is elemental composition, `z[species]` is charge,
and `nu[reaction, species, phase]` is signed stoichiometry.  A row that fails
either identity never reaches SymPy.

### Property records

Each chemical state needs data over a declared temperature and pressure
interval, with source provenance rather than one undocumented scalar:

- standard enthalpy, entropy, Gibbs energy, and heat-capacity coefficients;
- phase density or equation of state;
- vapour pressure or fugacity model and critical/triple-point data;
- latent heats, preferably derived from the same phase Gibbs functions;
- diffusivity, viscosity, conductivity, and surface tension where used;
- aqueous activity-model parameters and Henry constants;
- optical coefficients where rendering consumes the state;
- formation/reaction uncertainty and the interval in which coefficients are
  valid.

The repository's existing numbers should be imported with their existing
meaning and provenance.  They should not be silently promoted into a general
thermodynamic authority.

For the actual inorganic compendium, the appropriate upstream authorities are:

- NIST Chemistry WebBook SRD 69 for thermochemistry, phase transitions,
  vapour pressure, heat capacity, and fluid properties:
  https://webbook.nist.gov/
- NIST-JANAF SRD 13 for temperature-dependent thermochemical tables for more
  than 1,700 compounds:
  https://janaf.nist.gov/
- USGS PHREEQC databases for aqueous master species, explicit ions and
  complexes, mineral/gas phases, stoichiometry, temperature-dependent log K,
  surface species, exchange species, and activity models:
  https://www.usgs.gov/software/phreeqc-version-3

PHREEQC data is the right starting point for aqueous inorganic chemistry
because it already distinguishes master species, redox states, aqueous
species, pure phases, exchange species, and surface species.  NIST/JANAF is
needed beside it because the chamber also crosses cryogenic gas/liquid/solid
phase boundaries that an aqueous geochemistry database does not fully cover.

The compendium should be global, while a simulation build selects the closure
reachable from the elements, phases, and reactions actually admitted by its
ports and initial state.  Every active owner still receives the complete
active chemical-state axis.  This keeps identities fixed without compiling
thousands of unreachable compounds into a five-voxel dewar.

## State carried by every owner

The same inventory tensor governs all owners:

```text
n[owner, species, phase]          mol
T[owner]                          K
U[owner]                          J, preferred conserved thermal state
phase_volume[owner, phase]        m^3
```

Owner masks declare which phases are geometrically meaningful.  Masks prevent
invalid states; they do not change the inventory ABI.

### Voxels

Voxels admit gas, suspended liquid, suspended solid, and optionally plasma.
Rain, snow, cloud droplets, and aerosol populations are size/transport classes
of chemical inventories, not substitute chemical phases.  If those classes
must remain separate for settling, add a population axis:

```text
n[owner, population, species, phase]
```

The gas pressure is the sum of species fugacities.  There is no separate dry
air mass.  Nitrogen, oxygen, argon, carbon dioxide, water, and admitted trace
species all occupy the gas phase explicitly.

### Surfaces

A surface owner admits an aqueous/liquid film, one or more solid deposits, and
adsorbed states.  Film, frost, crust, and sediment become derived views over
composition and geometry.  Runoff transfers a composition vector and enthalpy,
not a water mass plus one salt scalar.

### Pools

A pool admits liquid/aqueous and solid phases.  It owns solvent and solute
amounts explicitly.  Volume, density, activity, freezing point, vapour
composition, precipitates, and overflow composition are derived.  Inflow and
overflow carry the same chemical-state span.

### Droplets and aerosols

A droplet owner carries its complete liquid/aqueous/solid composition plus
temperature, position, velocity, and geometric moments.  Radius follows from
phase volumes.  Dry radius is the solid residue volume, not one salt count.

An aerosol population carries number concentration and composition moments by
chemical state.  Activation transfers chemical inventory into a droplet
population conservatively.  Washout and deposition move that inventory to a
surface or pool owner.

### Ports and machine reservoirs

Every stream, bottle, vent, drain, and receiver transfers:

```text
chemical amount flux[species, phase]   mol/s
enthalpy flux                          W
momentum/pressure data                 existing fluid-port contract
```

Engine-toy may present kilograms, fill fractions, contaminant labels, and
fluid names, but those become views over this amount flux and compendium.  The
port can no longer manufacture or delete composition by editing only dry-air
and water columns after an accepted chamber step.

## Molarity, ions, bonding, and activities

For each owner with an aqueous phase:

```text
c_i = n_i / (1000 V_aq)                mol/L
b_i = n_i / m_solvent                  mol/kg solvent
I   = 1/2 sum_i(b_i z_i^2)             mol/kg
a_i = gamma_i(I, T, composition) b_i/b_standard
```

Molarity is required for state reporting, reaction rates written in molar
units, and volume-based transport.  Molality is required for thermodynamic
activities because it does not change merely through thermal expansion of the
solution.  Both must come from the same mole inventory.

At low and moderate ionic strength, a Davies or extended Debye-Hueckel model
can supply `gamma_i`.  Concentrated brines and cryogenic residual liquids need
Pitzer or SIT parameters where the compendium provides them.  The activity
model is selected by a declared parameter set and validity interval; silently
using ideal activity or Davies outside its range is not acceptable.

Each aqueous owner must publish and constrain:

```text
charge_balance = sum_i(z_i n_i) = 0
element_total[e] = sum_i(A[e,i] n_i)
ionic_strength
pH = -log10(a_H+)
```

Ionic bonding rules enter through the species/reaction table, charge balance,
activities, and electrochemical potentials.  This permits dissociation,
association, acid/base speciation, redox, complex formation, precipitation,
dissolution, adsorption, and ion exchange without inventing compounds from
names or charge signs.

## One combined SymPy system

### Conservation and transport

Let `B[edge, owner]` be the oriented geometry incidence matrix and
`J[edge, state]` the chemical-state molar flux.  Let `xi[owner, reaction]` be
reaction extents over one attempted dt.  The amount residual is:

```text
R_n = n_next - n_now
      - dt B.T @ J(n_next, T_next, P_next)
      - nu.T @ xi
      - dt source
```

The same paired flux is subtracted from its donor and added to its receiver.
Voxel faces, gas-to-film transfer, pool evaporation, runoff, precipitation,
droplet activation, settling, ports, drains, and machine reservoirs all use
this incidence form.  Conservation is therefore structural rather than a
diagnostic reconstructed afterward.

### Equilibrium and kinetics

For an equilibrium reaction:

```text
ln Q_r = sum_j(nu[r,j] ln a_j)
R_eq   = ln Q_r - ln K_r(T, P)
```

For a kinetic reaction integrated implicitly:

```text
rate_r = k_f(T) product(a_reactant ** order)
         - k_r(T) product(a_product ** order)
R_xi   = xi_r - dt rate_r(n_next, T_next)
```

Pure condensed phases use unit activity while present and complementarity at
appearance/disappearance.  Gas species use fugacity.  Aqueous species use
activity coefficients.  Mixed liquids use declared activity models.  Phase
transitions are ordinary equilibrium/kinetic rows between phase states, so the
same machinery governs water frost, CO2 deposition, liquid oxygen, salt
crystallization, and metal melting.

### Energy

One energy residual closes the same transition:

```text
R_U = U(n_next, T_next) - U(n_now, T_now)
      - dt Q_external
      - transported_enthalpy
      + sum_r(DeltaH_r(T) xi_r)
```

Formation enthalpies and phase enthalpies come from the compendium.  Latent
heat is the enthalpy difference between phase states.  Reaction heat, solution
heat, condensation heat, freezing heat, and port enthalpy therefore cannot be
counted twice or omitted by an ordering choice.

### Reduction into one step

The full unknown is a configured tensor containing next amounts,
temperatures/energies, reaction extents, and equilibrium variables.  SymPy
authors the residual `R(y)` and its Jacobian `dR/dy`.  One chamber step performs
a bounded damped Newton solve using the repository's tensor operations and
linear solve, then publishes one candidate state to the existing dt system.

The dt system remains unchanged.  It snapshots the participant, calls the
combined step, and accepts or restores the whole state.  The chemistry step
must refuse when Newton does not converge, a candidate becomes negative,
element or charge closure exceeds its declared error channel, a property model
is outside its validity interval, or energy fails to close.

This satisfies “one step” in the physical sense: all owners and processes
solve against the same candidate state.  Concatenating seven explicit calls
inside one Python function would retain the old split and does not satisfy the
contract.

The repository has `AbstractTensor.linalg.solve`, and fixed-count symbolic
Newton composition exists in other examples.  The dynamic coupled Jacobian
solve remains an implementation risk to test before generating the full
active closure.

### Measured implementation frontier, 2026-09-20

The tested implementation now lives under `src/common/chemistry`; the example
module is a compatibility import rather than a second copy.  `ChemistryScale`
defines positive characteristic variable and residual spans, converts physical
values to dimensionless `Precision[2]` coordinates with `AbstractTensor`, and
returns physical values without collapsing the expansion.  This is numerical
conditioning, not a second unit system or a replacement tensor backend.

`ReducedCompendium.compile_first_order_transition_piece` now builds and runs a
real `LLVMPiece` for the first bounded test case: one balanced reversible
first-order `A <-> B` transition with constant rate coefficients.  It solves
the reaction extent implicitly, updates one packed inventory span, retains two
limbs in the output, passes kernel-bank admission against its Python oracle,
contains `llvm.fma.f64`, and survives `LLVMPiece.save`/`load`.  The regression
uses two cells and verifies per-cell mole conservation.

The physical layout is one logical cell-major inventory
`n[cell, chemical_state]`, with two interleaved binary64 limbs per logical
entry.  Inputs that participate in wide arithmetic use the same layout.  The
output is one caller-owned span named `inventory_next`; this follows the
working compiled signal-kernel convention and avoids converting the state to
a tuple of scalar publications.

This measured piece does not yet claim the full chemistry step.  It explicitly
refuses equilibrium rows, multi-reactant stoichiometry, and nonzero Arrhenius
activation energies.  The remaining step is the bounded coupled Newton solve
over the reduced residual and energy equation, including the established
wide `sqrt`, `log`, and `exp` proof-core paths.  Metrics and structured
runaway publications also remain to be added before this piece can replace the
chamber participant.

A direct compiled probe established that a source spelling such as
`x[i].log()` inside a `Precision[2]` kernel is not itself a wide logarithm: no
precision pipeline receipt is produced for the call, and the kernel addresses
the physical limb slots as ordinary elements.  The coupled source must
therefore materialize the repository's proved range reduction and Horner cores
explicitly, as `signal_kernels.py` does.  Calling backend libm and widening its
answer afterward would not satisfy precision two.

There is also one measured runner integration requirement.  Precision-region
array operands are physical interleaved-limb spans, including the `dt` operand
once it participates in wide arithmetic.  `llvm_dt_system.py` currently fills
its generic `dt` column by broadcasting the ordinary scalar across the whole
span; doing that to a two-limb span would put the scalar in both high and low
lanes.  The piece therefore runs directly and persists today, but must not be
registered with that runner until the piece ABI declares limb width and the
runner writes high lanes while zeroing low lanes.

The chamber already owns the pause/freeze behavior in `CascadeMonitor`.  The
combined chemistry participant should feed that existing mechanism with
structured predicates evaluated after a converged candidate and before
acceptance: reaction power, temperature and pressure rates, reaction-rate
acceleration, depletion time, positivity/conservation failures, and property
validity.  A crossing causes dt refusal/refinement to the boundary and then
freezes the last accepted state.  Gameplay or scripting decides whether to
vent, quench, isolate, defrost, continue, or leave the event paused; event
detection does not alter the chemical equations.

## Publication and presentation contract

Every chemical publication needs structured identity:

```text
owner_id
species_id
phase_id
population_id, if applicable
quantity semantic
unit
span
```

Examples are `(voxel, H2O, liquid, cloud, concentration)` and
`(surface, CO2, solid, deposit, thickness)`.  LWC/IWC/RWC remain convenient
water views for the HUD and shader, while the renderer may sum or color any
declared species/phase optical contribution.  A bare `LWC` key is not an ABI.

The existing tensorized dt publication spans are the appropriate transport.
Species and phase order must be explicit compendium order, never dictionary
iteration or alphabetical reconstruction.

## Required migration

1. Add the provenance-bearing element/species/phase/reaction compendium and
   validate formula, mass, element, and charge identities at load time.
2. Import a documented initial atmospheric/cryogenic closure: N2, O2, Ar,
   CO2, H2O and their accessible gas/liquid/solid states, followed by the
   aqueous H/O/C/Na/Cl system needed for water, carbonate, and salt.
3. Define the canonical owner inventory and migrate constructors.  During the
   migration, old fields may be read-only views; there must be only one stored
   amount.
4. Author a small combined SymPy residual for one closed owner containing
   water autoionization, sodium-chloride dissolution, charge balance,
   molarity/molality/activity, and energy.  Verify it before adding geometry.
5. Add gas/liquid/solid phase rows for the atmospheric closure and verify a
   closed 77 K cell conserves every element, charge, and energy while producing
   physically allowed phases.
6. Add the owner-edge incidence system, first voxel-to-voxel and then
   voxel/surface/pool/droplet/port transfers.  Every transfer test must inspect
   equal and opposite species moles and enthalpy.
7. Replace `ChamberSim._advance_all` with one chemistry participant while
   retaining the existing package save/restore and managed-dt controller.
8. Convert engine-toy streams, dewar reservoirs, drains, and maintenance
   records to views and fluxes over the canonical inventory.
9. Replace bare publications and teach the HUD/shader to select or aggregate
   structured chemical channels.
10. Retire the independent water/salt/contaminant state only after round-trip
    conservation tests prove every existing scenario maps into the new state.

## Acceptance tests

The implementation is not complete until the following are observed:

- every reaction in the active compendium is element- and charge-balanced;
- zero-rate closed owners preserve every element, charge, and total energy;
- water autoionization and salt dissolution report molarity, molality, ionic
  strength, activities, pH, and electroneutrality from explicit ions;
- two salts with a common ion alter each other's activities and precipitation;
- an inflow mixture arrives in a voxel, surface film, pool, and droplet without
  losing any species or phase identity;
- runoff, overflow, settling, deposition, evaporation, and port flow move equal
  and opposite chemical amounts and enthalpy;
- at cryogenic temperature, N2/O2/Ar/CO2/H2O phase independently according to
  their chemical potentials and pressure rather than fixed temperature flags;
- oxygen enrichment emerges from preferential nitrogen transfer without an
  enrichment special case;
- the full candidate state rolls back after any rejected dt attempt;
- publications for two condensables cannot collide;
- hidden atmosphere rendering changes no chemistry state or cadence;
- the small combined law lowers and runs through the sanctioned native entry
  before a large compendium closure is attempted.

## Immediate architectural conclusion

The existing `voxel_species_step` should contribute its tested transport and
microphysical formulas to the combined residual, but its four water-like mass
categories cannot remain the state ABI.  The existing surface, pool, droplet,
salt, aerosol, and port records must all yield to the owner/species/phase mole
inventory.  The physical geometry remains where it is.  The chemistry table
states what matter is; owner geometry states where it is; the one SymPy
residual states how all of it may change together.
