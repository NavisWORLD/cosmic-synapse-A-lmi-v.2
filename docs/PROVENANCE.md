# Provenance and Restoration Record

This repository preserves a long-running research and software lineage. The 2026-09-14 restoration is intentionally additive: it does not rewrite Git history, delete historical generations, or replace theory artifacts with a clean-room rewrite.

## Preservation boundary

Before restoration work began, the active repository head was:

`1770061052f7ac92292f7a27e84df874a4a31098`

That state is preserved on:

`preservation/pre-restoration-2026-09-14`

Restoration work is isolated on:

`restoration/complete-system-2026-09-14`

Draft review is tracked in pull request #1.

## Restoration method

The restoration uses a red/green contract sequence wherever practical:

1. Preserve the pre-restoration state.
2. Inspect source behavior and identify a concrete seam.
3. Add a deterministic contract that fails on the existing behavior.
4. Repair the smallest active implementation boundary.
5. Run the expanded suite in GitHub Actions.
6. Keep optional-service/hardware claims separate from deterministic software results.

The failed runs are part of the evidence trail; they are not erased or reclassified as passes.

## Major restoration checkpoints

The branch contains staged repairs covering:

- configuration loading and startup contracts;
- password-encryption round trips and derivation metadata;
- LightToken spectral representation and vector-schema consistency;
- raw-artifact persistence and provenance;
- explicit multimodal embedding spaces;
- lazy optional imports for audio/ML/infrastructure/visualization stacks;
- Neo4j graph loading/visualization boundaries;
- deterministic CST state and replay interface;
- hypothesis provenance/uncertainty;
- federated-learning terminology and reproducible experimental noise;
- HRCS packet/crypto/acoustic/mesh/simulation/radio contracts;
- versioned Python/Unity IPC;
- root packaging, CLI diagnostics, clean wheel installation;
- God Music deterministic tests and production web build;
- local Docker Compose hardening and credential/port contracts.

A fully green code/config checkpoint after the Compose contract repair is:

`cdbd5b9c0cb5f2071437e33cbf5a89241881e8d2`

GitHub Actions run #102 for that commit completed successfully.

Later commits in the same branch update active documentation and metadata while preserving that implementation boundary.

## Historical material

Historical artifacts remain intentionally present, including older HTML demos, publications, PDFs, ZIP archives, CST terminology, God Music generations, Unity material, and communications experiments.

Examples visible in the preserved repository include:

- `The Cosmic Synapse Madsens theory.pdf`
- `The-theory-of-CST-main (2).zip`
- `Harmonic_Resonance_AI_Music_Conductor_Complete_Publication.md`
- `Harmonic_Resonance_AI_Music_Conductor_Complete_Publication.md.pdf`
- `12D_Cosmic_Synapse_Audio_Engine-demo.html`
- `ULTIMATE_AI_Band_Conductor_v4_Complete.html`
- `god music/`
- `cosmic_synapse/Unity/`
- `coms/hrcs/`

Their presence is evidence of project history, not automatic validation of every claim made inside them.

## Active-vs-historical rule

The root README and the files under `docs/` define the current engineering/evidence boundary. Historical documents are preserved as authored unless a file is part of the active user/developer path and would otherwise mislead a new user about installation, security, test status, or scientific validation.

When active documentation conflicts with a historical claim, use the active documentation for current software status and use Git history/preservation branch to inspect the original statement.

## Authorship

Repository authorship metadata identifies Cory Shane Davis as the project author/maintainer in the active package metadata. This provenance record does not attempt to adjudicate priority claims beyond what the repository history itself can demonstrate.

## Non-claims

The restoration record does not convert repository history into proof of:

- consciousness or sentience;
- AGI;
- biological life;
- a new physical law;
- additional physical dimensions;
- quantum advantage or quantum consciousness;
- golden-ratio superiority;
- formal differential privacy;
- cryptographic secure aggregation;
- radio anti-jamming superiority;
- hardware performance not measured in a controlled test.

Those require evidence beyond preserving code, commits, and deterministic software tests.
