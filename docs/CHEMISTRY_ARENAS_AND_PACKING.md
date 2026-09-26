# Chemistry arenas and tensor packing

## Contract

One reduced chemical compendium defines one ordered state axis. Every material
owner uses that order. An owner may disable phases and reaction families, but
it does not reinterpret a state slot. This gives the compiled law one stable
ABI across gas voxels, droplets, films, liquid pools, ports, machine fluids,
solid surfaces, and metal volumes.

The physical inventory span is flattened in this order:

```text
[owner][chemical state][precision limb]
```

Owners are grouped into contiguous arena blocks. This permits a kernel to march
one arena densely while the global owner number remains authoritative at every
boundary. Precision limbs are adjacent and high-limb first, matching
`ChemistryPieceLayout` and the existing chemistry `LLVMPiece` boundary.

Owner metadata is tensorized beside the inventory:

- arena identity, arena kind, and local owner index;
- a phase activity matrix `[owner, phase]`;
- the phase index of each state and a directly materialized eligibility matrix
  `[owner, state]`;
- a reaction-family activity matrix `[owner, family]`;
- temperature, internal energy, pressure, and configured phase volumes as
  owner state supplied by the engine and managed-dt caller.

Boundary metadata is also tensorized:

- left and right global owner numbers;
- boundary kind;
- a transport-family matrix `[boundary, family]`;
- a boundary-reaction matrix `[boundary, family]`;
- geometry values such as area, metric distance, permeability, and transfer
  coefficients supplied with the configured machine geometry.

A positive boundary flux moves inventory from its left owner to its right
owner. The scatter operation subtracts and adds the identical value, so
transport cannot create or destroy a chemical state. Reaction rows then change
states within an owner or at a declared interface, subject to the compendium's
element and charge conservation checks. Heat is transported on the same owner
and boundary identities, while its constitutive law may come from the existing
Laplace/metric-tensor system.

## Dewar arenas

`dewar_chemistry_plan` describes the coupled simulations already present in
the chamber and engine machine model.

### Chamber gas

Each chamber voxel is a gas owner. Rectilinear face pairs provide conservative
advection, diffusion, and energy exchange. The builder accepts any positive
three-dimensional shape; `(5, 5, 5)` produces 125 gas owners. A future column
geometry can produce a different face table without changing the chemical
state ABI.

### Aerosol and droplet population

Each gas voxel has a matching population owner for liquid droplets, ice, and
other carried condensed matter. Gas-to-population boundaries host nucleation,
droplet growth, evaporation, and freeze/thaw families. Droplet composition is
therefore inventory in the same state system, rather than an unlabelled water
mass.

### Pool and drain

The bottom condensate pool is an aqueous/liquid/solid owner. Bottom gas faces
exchange species and heat with it. A directed fluid-line boundary connects the
pool to the drain receiver, so a blocked or frozen drain changes boundary
conductance rather than bypassing chemistry. Both pool and receiver retain
their complete species mixtures.

### Inner walls and cold tip

The five closed inner vessel panels have separate reactive-surface owners and
solid-bulk owners. Gas/surface boundaries carry adsorption, condensation,
corrosion, and deposition. Surface/bulk boundaries carry heat and solid-state
diffusion.

The cold tip uses the same two-layer form: a reactive exposed surface in the
top-center gas voxel and a copper bulk owner behind it. Frost and deposits live
on the surface. Oxidation, diffusion, and metal phase changes live in the bulk
or on its declared surface boundary. This is the chemistry counterpart of the
engine machine's protruding interaction surface.

### Cold-head process fluid

The working-gas supply and return passages are fluid owners. They connect to a
regulated reservoir through supply and return line boundaries and to the cold
tip through energy-transfer boundaries. The process fluid can therefore carry
a real species mixture and phase inventory while the machine supplies the
pressure differential, compressor work, recuperator conditions, and line
conductance.

### Open atmosphere port

The outside atmosphere is a reservoir owner connected to the top-center gas
voxel by an open-port boundary. Pressure-flow, species, and energy transport
all cross the same declared interface. Chamber pressure is consequently a
state derived from inventory, volume, and temperature; it is not an isolated
display condition.

### Vacuum annulus

The insulation vacuum is deliberately not a chemical voxel arena. It is a
scalar gap with pressure, integrity, and thermal conductance. The engine owns
that state. It modifies heat transfer between the inner and outer shells, but
allocates no per-voxel chemical inventory. If material leakage into the annulus
is later required, it must be represented as a material owner connected by a
leak boundary; it must not silently turn the insulation gap into chamber gas.

## Solid-surface reaction arena

`solid_surface_reaction_plan` provides the general three-layer arrangement:

```text
environment fluid <-> reactive surface <-> solid substrate
```

The environment may be gas, aqueous fluid, or liquid. The surface owns
adsorbates, deposits, corrosion products, and surface liquid. The substrate
owns bulk solid inventory. The first interface admits adsorption, oxidation,
corrosion, deposition, and dissolution; the second admits solid diffusion,
phase transformation, and energy transfer. Multiple patches remain contiguous
and use one-to-one oriented boundary rows.

## Metallurgy arena

`metallurgy_chemistry_plan` provides five coupled blocks:

```text
furnace atmosphere -> metal surface -> metal bulk
                                      <-> grain boundary
                                      <-> slag
```

Metal bulk and grain-boundary owners use the metallurgy arena kind. Their
reaction masks distinguish ordinary solid diffusion, fast grain-boundary
diffusion, alloying, precipitation, and phase transformation. The external
surface hosts adsorption, oxidation, deposition, and dissolution. The slag is
a liquid/solid fluid arena with dissolution, precipitation, and slag-reaction
families. The same layout extends to weld pools, heat-treatment furnaces,
electrodes, refractory interfaces, and catalytic beds by adding arena and
boundary rows rather than adding a second chemistry engine.

## Deployment sequence

`deployment_manifest` determines the species limit before any tensor is
allocated. Each configured component declares the exact chemical states it
initially contains. The compendium then walks its reaction hypergraph to a
fixed point. A reaction contributes its other states only when all reactants
for an enabled direction are already reachable. Reversible equilibria may be
entered from either side; irreversible kinetic rows only expand in a direction
whose prefactor is nonzero. This includes chained products and phase
transitions while excluding unrelated compounds that merely use the same
elements.

The resulting manifest contains:

- the authoritative ordered chemical-state axis and derived capacity;
- the ordered species identities represented by those states;
- the active reactions, phases, and families;
- a SHA-256 ABI identity for the complete configuration.

The dewar packing plan is built from `manifest.states`.
`EngineSpeciesExchangeLayout` uses the same manifest for every machine-side
endpoint and allocates `[endpoint, state, precision limb]` in moles. It converts
engine mass dictionaries through the compendium's molar masses and rejects a
state outside the manifest instead of truncating or silently growing only one
side. The engine's mixture dictionaries already have no fixed numeric ceiling;
the manifest supplies the canonical phase-qualified identities and exact
capacity they previously lacked.

The capacity is arbitrary in the sense required here: it is derived from the
configured compendium and component closure, with no water/air list or numeric
maximum in the packing or exchange code. Changing the components or adding a
reachable transition produces a new manifest, capacity, and ABI identity. A
running compiled deployment remains fixed-size; changing that closure requires
a new reduction and compilation because its law and state ABI have changed.

The packing code establishes identity and conservation. The full deployment
still has these explicit stages:

1. Reduce the master SymPy compendium once for the configured machine,
   elements, phases, and enabled reaction families.
2. Materialize the owner and boundary tables from the machine geometry and
   port graph.
3. Assemble boundary transport deltas and owner/interface reaction residuals
   into one dimensionless `Precision[2]` system.
4. Lower that system through the existing sanctioned compiler entry into one
   chemistry `LLVMPiece`.
5. Register the packed inventory and thermodynamic columns with the existing
   managed-dt system. Accepted engine ticks commit machine/chamber exchange;
   rejected ticks restore those same columns.

The current implementation completes the first packing layer, its dewar and
general surface/metallurgy topology builders, and conservative boundary
scatter. It does not claim that the complete nonlinear compendium has already
been lowered or joined to the engine tick.
