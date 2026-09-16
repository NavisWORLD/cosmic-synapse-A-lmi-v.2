# Python / Rust Interoperability

Python is the reference implementation for this closure. Rust is required to consume and produce the same active portable contracts without weakening validation.

## `.cosmos` directions under test

```text
Python workspace -> Python export -> .cosmos -> Rust verify -> Rust import
Rust workspace   -> Rust export   -> .cosmos -> Python verify -> Python import
```

The interoperability test suite creates only synthetic data. Both implementations must accept the other implementation's bundle and preserve workspace identity and deny-by-default authority.

## Deterministic archive rules

The active format uses canonical sorted compact JSON and a trailing newline for generated JSON files. Bundle payloads are sorted. ZIP metadata is controlled so repeated exports of an unchanged workspace are byte reproducible. SHA-256 addresses bundle and payload integrity.

Interoperability tests require repeated exports from each implementation to be byte-identical when the workspace is unchanged.

## CST state parity

Cross-language CST replay begins from the persisted canonical `state/cst.json` envelope. The persisted phase is authoritative. A language/provider swap must not regenerate an existing state from a seed using a different language's PRNG.

Given the same persisted state and event sequence, Python and Rust snapshots are compared numerically across `x12`, `m12`, `omega`, `phase`, `energy`, `entropy`, and `step`.

This is deterministic computational-state interoperability. It is not evidence of physical extra dimensions or new physics.

## Compatibility boundary

Unknown workspace, bundle, component, CST, or ABI versions are errors. Compatibility is established by explicit tests and golden/synthetic fixtures, not by permissive partial deserialization.
