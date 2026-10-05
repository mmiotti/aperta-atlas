"""Opt-in dependency tracking for `aperta_atlas` scripts.

When `APERTA_TRACK_DEPENDENCIES=1` in `.env`, every script run records:

  - which input data files it read (`used_data`)
  - which output data files it wrote (`created_data`)
  - which config / variant parameters it actually used
  - a content-aware hash of the script source (`get_file_hash`)

Subsequent runs validate that upstream data hasn't gone stale — the
hash of the script that produced it must still match. Warnings fire on
mismatches; the tracker never aborts.

State lives in `status/status.json` at the project root, keyed by
`<script_name>/<scenario>/<variant>`. The data shape is the dict
returned by `get_current_status(context)` per run.

This module holds the *implementation* — the I/O on `status.json`,
hashing, status-tree lookup, dependency validation, and the top-of-run
banner. The `Context` class in `aperta_atlas.context` owns the *state*
(the `created_data` / `used_data` sets, the `track_dependencies` flag)
and delegates here for the actual work. Functions here all take a
`Context` as their first argument.
"""

import datetime
import hashlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aperta_atlas.context import Context


# ---------------------------------------------------------------------------
# Hashing — content-aware for .py files so cosmetic edits don't invalidate
# ---------------------------------------------------------------------------


def _normalize_python_source(file_path: str) -> bytes | None:
    """Return tokens-only bytes for a .py file (comments + docstrings + inter-token
    whitespace stripped) suitable for hashing.

    Keeps semantically meaningful structure (identifiers, non-docstring literals,
    operators, indentation level via INDENT/DEDENT tokens, statement boundaries via
    NEWLINE) so the resulting hash is stable across cosmetic edits — adding a
    comment, editing a docstring, reflowing whitespace, deleting blank lines — but
    flips on any real code change.

    Docstrings are identified via AST: a string literal that is the first
    expression statement of a Module, FunctionDef, AsyncFunctionDef, or ClassDef
    body. Other string literals (values, stringified type annotations, multi-line
    SQL constants, etc.) are kept in the hash.

    Returns `None` on parse/tokenize failure so the caller falls back to a raw-
    bytes hash for that file.
    """
    import ast
    import tokenize
    from io import BytesIO
    try:
        with open(file_path, 'rb') as f:
            source = f.read()
        tree = ast.parse(source)
        # Track the (line, col) of each docstring's STRING token so we can skip it
        # (and its trailing NEWLINE) during tokenization. Position-keyed lookup is
        # unambiguous — STRING tokens emit at the same (lineno, col_offset) as the
        # AST node for their value.
        docstring_positions: set[tuple[int, int]] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    v = node.body[0].value
                    docstring_positions.add((v.lineno, v.col_offset))

        skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
        parts: list[str] = []
        drop_next_newline = False
        for tok in tokenize.tokenize(BytesIO(source).readline):
            if tok.type in skip:
                continue
            if tok.type == tokenize.STRING and tok.start in docstring_positions:
                # Drop the docstring STRING and the NEWLINE that ends its expression
                # statement, so a function with and without a docstring tokenize-equal.
                drop_next_newline = True
                continue
            if drop_next_newline and tok.type == tokenize.NEWLINE:
                drop_next_newline = False
                continue
            drop_next_newline = False
            if tok.type == tokenize.INDENT:
                parts.append('<I>')
            elif tok.type == tokenize.DEDENT:
                parts.append('<D>')
            elif tok.type == tokenize.NEWLINE:
                parts.append('<NL>')
            else:
                parts.append(tok.string)
        return ' '.join(parts).encode('utf-8')
    except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError, ValueError, FileNotFoundError):
        return None


def get_file_hash(file_path: str) -> str:
    """MD5 hash of a file. For `.py` files, comments and superfluous whitespace
    are excluded so cosmetic edits don't invalidate dependency tracking; for any
    other file (or on tokenization failure) the raw bytes are hashed.
    """
    try:
        if file_path.endswith('.py'):
            normalized = _normalize_python_source(file_path)
            if normalized is not None:
                return hashlib.md5(normalized).hexdigest()
        with open(file_path, 'rb', buffering=0) as f:
            return hashlib.file_digest(f, 'md5').hexdigest()
    except FileNotFoundError:
        logging.warning(f"Could not hash file: not found ({file_path})")
        return 'none'


def get_timestamp() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp())


# ---------------------------------------------------------------------------
# status.json I/O
# ---------------------------------------------------------------------------


def read_status(context: 'Context') -> dict[str, dict]:
    file_path = f'{context.working_dir}/status/status.json'
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        logging.warning(
            f"status.json at {file_path} is empty or malformed; treating as "
            f"no prior state. Prior tracking data will be lost on next write.")
        return {}


def write_status(context: 'Context', status: dict[str, dict]) -> None:
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, type):
                return str(obj)
            return json.JSONEncoder.default(self, obj)
    file_path = f'{context.working_dir}/status/status.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2, cls=_Encoder)


# ---------------------------------------------------------------------------
# status-tree navigation
# ---------------------------------------------------------------------------


def get_status_from_dict(
    status: dict, script_name=None, scenario_name=None, variant_name=None,
) -> dict:
    res = status
    if script_name:
        res = res.get(script_name, {})
    if scenario_name:
        res = res.get(scenario_name, {})
    if variant_name:
        res = res.get(variant_name, {})
    return res


def find_related_status(initial_status: dict, context: 'Context') -> dict:
    """Locate a prior-run status entry related to this run.

    Tries (in order): exact script + scenario + variant match; any variant of
    the same scenario; any scenario + variant of the same script. Returns the
    deepest match found, or an empty dict.
    """
    s = get_status_from_dict(
        initial_status, context.status_script_name, context.status_scenario_name,
    )
    if s:
        return s[next(iter(s))]
    s = get_status_from_dict(initial_status, context.status_script_name)
    if s:
        s = s[next(iter(s))]
        return s[next(iter(s))]
    return {}


def get_status_of_data_creator(status: dict, data_name: str) -> tuple[dict, int]:
    """Return (creator_status, n_creators) for the script that registered `data_name`
    as a created output. Multiple creators is a misconfiguration — the warning is
    surfaced by `check_dependencies`."""
    found = 0
    result = {}
    for _script_name, scenarios in status.items():
        for _scenario_name, runs in scenarios.items():
            for _variant_name, sstatus in runs.items():
                if data_name in sstatus['created_data']:
                    result = sstatus
                    found += 1
    return result, found


# ---------------------------------------------------------------------------
# Status update at end-of-run
# ---------------------------------------------------------------------------


def get_current_status(context: 'Context') -> dict:
    """Snapshot of this run's tracked state, ready to write to status.json."""
    return {
        'script_name': context.status_script_name,
        'scenario_name': context.status_scenario_name,
        'variant_name': context.status_variant_name,
        'timestamp': get_timestamp(),
        'runtime': time.perf_counter() - context.start_time,
        'file_path': context.caller_file_path,
        'file_hash': get_file_hash(
            os.path.join(context.caller_root_path, context.caller_file_path),
        ),
        'used_variant': (
            context.variant.used_fields_as_dict() if context.variant is not None else None
        ),
        'used_data': list(context.used_data),
        'created_data': list(context.created_data),
    }


def update_status(context: 'Context') -> dict:
    """Read existing status.json, merge in this run's snapshot, write back."""
    status = read_status(context)
    s = get_current_status(context)
    status.setdefault(
        s['script_name'], {},
    ).setdefault(s['scenario_name'], {})[s['variant_name']] = s
    write_status(context, status)
    return status


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------


def check_dependencies(
    context: 'Context', data_names: list[str], initial_check: bool,
) -> bool:
    """Validate that each entry in `data_names` was created by a script whose
    current hash and used-config still match what was recorded at creation time.

    `initial_check=True` runs once at script start over the full prior `used_data`
    list. `initial_check=False` runs per `register_used_data` call to catch
    newly-read data that wasn't seen at start. Logs warnings on mismatch; never
    aborts.
    """
    this = context.parent if context.parent else context
    if not initial_check:
        ignore = get_status_from_dict(
            this.initial_status, this.status_script_name,
            this.status_scenario_name, this.status_variant_name,
        ).get('used_data', [])
    else:
        ignore = []

    count = 0
    # Split by severity: "no creator" is often benign on first runs — NOTE.
    # Real staleness (multiple creators; upstream script edited since output) — WARNING.
    notes_list: list[str] = []
    warnings_list: list[str] = []
    creators: dict[str, dict] = {}

    for data_name in data_names:
        if data_name in ignore:
            continue
        creator_status, found = get_status_of_data_creator(this.initial_status, data_name)
        if found == 0:
            notes_list.append(f"No creator registered for file `{data_name}`.")
            continue
        if found > 1:
            warnings_list.append(
                f"More than one creator registered for `{data_name}`. "
                f"Consider flushing status.json.",
            )
        key = (
            creator_status['script_name']
            + creator_status['scenario_name']
            + creator_status['variant_name']
        )
        creators[key] = creator_status

    for cstatus in creators.values():
        if get_file_hash(
            os.path.join(this.caller_root_path, cstatus['file_path']),
        ) != cstatus['file_hash']:
            warnings_list.append(
                f"File `{cstatus['script_name']}` has been updated since output was created.",
            )
        count += 1

    char = "↳" if not initial_check else " "
    for n in notes_list:
        logging.note(f" {char}  Dependency note: {n}")
    if warnings_list:
        for w in warnings_list:
            logging.warning(f" {char}  Dependency warning: {w}")
    elif count > 0:
        logging.info(
            f" {char} "
            f"{'✅ ' + str(count) + ' dependencies validated.' if initial_check else '✅ Dependency validated.'}"
        )
    return True


# ---------------------------------------------------------------------------
# Banner / log header at script start
# ---------------------------------------------------------------------------


def _param_value(param) -> str:
    if not isinstance(param, (float, int, str, type)):
        return f'{type(param)}...'
    if isinstance(param, int):
        return f'{param:,}'
    return str(param)


def log_context_info(context: 'Context') -> None:
    """Print the start-of-run banner: namespace, project, variant, and (when
    tracking is on) prior-run status + a dependency-validation sweep."""
    logging.info("=" * 91)
    vasuffix = f' / {context.status_variant_name}' if context.variant else ''
    logging.info(f"Starting `{context.caller_base_name}`{vasuffix}")
    logging.info(f"   Namespace:     {context.namespace}")
    if context.role == 'project':
        logging.info(f"   Project:       {context.project} | Scenario: {context.scenario}")
    if context.variant:
        logging.info(
            f"   Variant param: "
            f"{', '.join(f'{p}={v}' for p, v in context.variant._asdict().items())}",
        )
    if not context.track_dependencies:
        logging.info("=" * 91)
        return
    # Tracking on: pull prior-run status + verify dependencies.
    initial_status = get_status_from_dict(
        context.initial_status, context.status_script_name,
        context.status_scenario_name, context.status_variant_name,
    )
    related = False
    if not initial_status:
        initial_status = find_related_status(context.initial_status, context)
        related = True
    if not initial_status:
        logging.info("   Script status not available. Likely first run.")
    else:
        if related:
            sa = (
                initial_status['scenario_name']
                if initial_status['scenario_name'] != context.status_scenario_name else ''
            )
            sb = (
                initial_status['variant_name']
                if initial_status['variant_name'] != context.status_variant_name else ''
            )
            logging.info(f"   ** Inferred from: `{' / '.join(s for s in (sa, sb) if s)}` **")
            spaces = "   "
        else:
            spaces = ""
        # `used_config` log line removed — see comment above. Old
        # `status.json` entries may still carry the field; ignored.
        logging.info(f"{spaces}   Prev. runtime: {initial_status['runtime']:.1f}s")
        if not related:
            check_dependencies(context, initial_status['used_data'], True)
    logging.info("=" * 91)
