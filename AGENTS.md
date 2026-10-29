# Agent Instructions

This file is the entry point for coding agents working in this repository. Keep
changes small, inspect the existing code before editing, and follow the
repository-specific rules in `.agent/`.

## Rule Files

- Read `.agent/design_principles.mdc` before non-trivial code changes.
- Use `.agent/python_rules.mdc` and `.agent/package_format_rules.mdc` for Python
  code.
- Use `.agent/cpp_rules.mdc` for C++, CUDA, and Warp-adjacent native code.
- Use `.agent/tests.mdc` before adding or changing tests.
- Use `.agent/sphinx_md.mdc`, `.agent/documenter.mdc`, and
  `.agent/diataxis.mdc` for documentation changes.
- Use `.agent/file_naming.mdc` when creating or moving implementation files.

## Development Workflow

- Prefer `rg`/`rg --files` for repository search.
- Activate `.venv` before running Python commands when it exists.
- Preserve user changes in the working tree. Do not revert unrelated files.
- Keep public APIs minimal and user-focused. Do not expose implementation details
  unless they are required by the user-facing contract.
- Prefer existing local patterns and helpers over new abstractions.
- For new NVIDIA-authored files, use
  `SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
  followed by `SPDX-License-Identifier: Apache-2.0`. Preserve existing copyright
  years and upstream notices when modifying older files.
- Add focused tests for behavioral changes. Do not use `float64` in tests.
- Keep CUDA graph enabled for normal runtime, examples, and performance-sensitive
  paths. Disable CUDA graph only in tests or debugging paths where eager
  execution helps catch correctness issues.

## Python Conventions

- Add type hints to all function signatures.
- Use concise docstrings that describe behavior, inputs, outputs, and tensor
  shapes when relevant.
- For dataclasses, document fields where they are declared.
- Use logging/error helpers from `curobo/_src/util/logging.py` instead of raw
  prints or ad hoc exceptions in library code.
- Keep `_src` package initializers marker-only unless they intentionally build a
  public API aggregation layer.
- Public modules under `curobo/` should curate user-facing imports. Internal
  code should import concrete implementation modules directly.

## Naming

- Internal implementation files under `curobo/_src/` should use the
  category-first pattern from `.agent/file_naming.mdc`.
- Public API modules under `curobo/` should use descriptive, user-friendly names.

## Documentation

- Prefer Markdown with MyST syntax for new Sphinx documentation.
- Keep docs in the correct Diataxis category: tutorial, guide, concept, or
  reference.
- Write user-facing docs around the outcome and contract, not internal
  implementation details.
