# AbstractUI color selectors, inventories, and active tools

A color selector is a specialization of the canonical `input` primitive. It
declares `set-color`, a destination identity, and a color value. It does not
assume an HTML `<input type=color>`; browser, controller, textual, and world
backends may realize the same semantic input differently.

An entity inventory is an identified collection of references to entities. An
inventory item may declare that its entity is usable as a tool. Equipping it
creates an `ActiveTool` relation among inventory, item, and entity. Removing an
equipped item clears that relation rather than leaving a dangling tool.

These records do not require entities to have originated from archetypes. An
archetype can dispense or describe an entity, but the inventory only requires
an identified entity reference. This keeps archetypes in their intended role:
a universe of reusable recipes, not a compulsory ontology gate.
