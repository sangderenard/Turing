# Checkpoint: modular per-light accessory contract

Status: design checkpoint only, not a published or implemented attachment API. The load-system release has priority. Current tested can seal is live; fixed-underlight configurable grid UI remains source-only and must not be released as the final modular system.

The owner of light attachment authoring will use one ordered collection of stable instances, each naming a light, type, enabled state, local optical-axis offset and type-specific settings. Proposed envelope: version1 plus attachments [{id, lightId, kind, enabled, offset, settings}]. Names and exact validator are to be finalized with optical/material owner before code. Keep one authoritative saved definition; no parallel UI-only state. Settings appear under their light only when attached. Retain explicit shape compatibility instead of showing nonfunctional controls.

Existing grid/snoot values migrate to attached tools without resizing or re-aiming saved rigs. Mechanical adapters reuse actual geometry; grid footprint follows emitter/can, cell count determines pitch, and apron seals to the can. Snoot needs real length and exit aperture geometry. Do not duplicate existing kicker/stage tubes. No independent grid width that can drift from the source.

Tools reference the existing stage/light material engine rather than copied independent optical coefficients. Optical lenses must use the original centrally maintained parametric lens definitions and their actual material/travel-distance volume path. No thin substitute pretending to be that system. Alpha cookies need a real mask/transport contract, not only a Phong texture.

Packing hazard: current fixture faces use local emitter IDs and one local housing ID, while room/table/gobo faces use the shared catalogue remapper. Material-bearing tool faces must enter the shared catalogue path (or an equivalent explicit remap), never reuse numeric catalogue IDs as emitter-local IDs. That collision could accidentally make an accessory emissive. Shared geometry must remain identical between Phong and ray views, with A/B separation and actual extent checks.
