# Rust Backend

This directory contains a prototype Rust implementation for accelerated tensor operations.

- `Cargo.toml` defines a `cdylib` crate built with `pyo3`.
- `src/lib.rs` exposes minimal stub functions.
- `setup_env.sh` installs the optional `rust` dependency group via the repository's setup script.

Follow the coding standards from the parent directory. The Rust library is expected to be loaded by `rust_backend.py`.
