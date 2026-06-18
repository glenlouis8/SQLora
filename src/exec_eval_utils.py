"""
exec_eval_utils.py
──────────────────
Execution accuracy for text-to-SQL evaluation.

Pipeline per sample:
  1. Parse CREATE TABLE statements from row["context"]
  2. Generate synthetic rows (type-aware, seeded)
  3. Build in-memory SQLite DB
  4. Execute reference SQL  → ground truth result set
  5. Call Modal API         → predicted SQL
  6. Execute predicted SQL  → predicted result set
  7. Compare (order-agnostic) → correct or not
"""

import re
import sqlite3
from typing import Optional

import numpy as np
import requests

try:
    import sqlglot
    import sqlglot.expressions as exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


# ── Synthetic data pools ──────────────────────────────────────────────────────

_INTS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 500, 1000, 9999]
_TEXTS = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Sales", "Marketing", "Engineering", "Finance", "Operations",
    "Active", "Inactive", "Pending", "Done", "Draft",
]
_FLOATS = [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 9.99, 10.0, 15.5, 19.99, 20.0, 25.0, 49.99, 50.0, 99.99, 100.0, 199.99, 500.0, 999.99, 1999.99]
_DATES = [
    "2020-01-01", "2020-06-15", "2021-03-10", "2021-09-20", "2022-01-01",
    "2022-07-04", "2023-01-15", "2023-06-30", "2023-12-01", "2024-01-01",
    "2024-03-15", "2024-06-01", "2024-09-30", "2024-12-31", "2025-01-01",
    "2025-03-01", "2025-06-01", "2025-09-15", "2025-11-01", "2025-12-31",
]
_DATETIMES = [d + " 00:00:00" for d in _DATES]


def _infer_type_category(type_str: str) -> str:
    t = type_str.upper()
    if any(x in t for x in ["DATETIME", "TIMESTAMP"]):
        return "datetime"
    if "DATE" in t:
        return "date"
    if any(x in t for x in ["FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC", "MONEY"]):
        return "float"
    if any(x in t for x in ["INT", "NUMBER", "SERIAL", "BIGSERIAL"]):
        return "int"
    if any(x in t for x in ["BOOL"]):
        return "int"
    return "text"


def _pool_value(type_cat: str, row_idx: int, col_idx: int):
    idx = (row_idx * 7 + col_idx * 3) % 20
    if type_cat == "int":
        return _INTS[idx % len(_INTS)]
    if type_cat == "float":
        return _FLOATS[idx % len(_FLOATS)]
    if type_cat == "date":
        return _DATES[idx % len(_DATES)]
    if type_cat == "datetime":
        return _DATETIMES[idx % len(_DATETIMES)]
    return _TEXTS[idx % len(_TEXTS)]


# ── WHERE value extraction ────────────────────────────────────────────────────

def extract_where_values(sql: str) -> dict:
    """
    Parse WHERE clause and return {col_name_lower: value_to_inject}.

    Handles = (inject exact value), > (inject val+1), < (inject val-1),
    >= (inject exact), <= (inject exact). String values injected as-is.
    Only works with sqlglot; returns {} on failure.
    """
    result = {}
    if not HAS_SQLGLOT:
        return result
    try:
        # Try MySQL dialect first — dataset uses double-quoted strings (MySQL style)
        stmt = None
        for dialect in ("mysql", None):
            try:
                stmt = sqlglot.parse_one(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
                if stmt is not None:
                    break
            except Exception:
                continue
        if stmt is None:
            return result

        op_map = {
            exp.EQ: "eq",
            exp.GT: "gt",
            exp.LT: "lt",
            exp.GTE: "gte",
            exp.LTE: "lte",
        }

        for node in stmt.walk():
            op_type = type(node)
            if op_type not in op_map:
                continue

            left, right = node.left, node.right

            # Normalize: column always on left
            if isinstance(right, exp.Column) and not isinstance(left, exp.Column):
                left, right = right, left
                # flip comparison direction
                flip = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ}
                op_type = flip.get(op_type, op_type)

            if not isinstance(left, exp.Column):
                continue
            if not isinstance(right, (exp.Literal, exp.Neg)):
                continue

            col = left.name.lower()
            is_string = isinstance(right, exp.Literal) and right.is_string

            if is_string:
                val = str(right.this)
                result[col] = val
            else:
                raw = right.this if isinstance(right, exp.Literal) else f"-{right.this.this}"
                try:
                    num = float(raw) if "." in str(raw) else int(raw)
                except (ValueError, TypeError):
                    continue
                if op_type == exp.GT:
                    result[col] = num + (1 if isinstance(num, int) else 0.1)
                elif op_type == exp.LT:
                    result[col] = num - (1 if isinstance(num, int) else 0.1)
                else:  # EQ, GTE, LTE
                    result[col] = num

    except Exception:
        pass
    return result


# ── Schema parsing ────────────────────────────────────────────────────────────

def _parse_schema_regex(context: str) -> list[dict]:
    tables = []
    table_pat = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)[`"\]]?\s*\(([^;]+?)\)',
        re.IGNORECASE | re.DOTALL,
    )
    col_pat = re.compile(r'^\s*[`"\[]?(\w+)[`"\]]?\s+(\w+(?:\s*\([^)]*\))?)', re.IGNORECASE)
    skip_keywords = {"PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "KEY", "CONSTRAINT", "CHECK"}

    for m in table_pat.finditer(context):
        table_name = m.group(1)
        col_block = m.group(2)
        columns = []
        for line in col_block.split(","):
            line = line.strip()
            if not line:
                continue
            first_word = line.split()[0].upper().strip('`"[]')
            if first_word in skip_keywords:
                continue
            cm = col_pat.match(line)
            if cm:
                columns.append({"name": cm.group(1), "type": cm.group(2)})
        if columns:
            tables.append({"name": table_name, "columns": columns})
    return tables


def parse_schema(context: str) -> list[dict]:
    if not HAS_SQLGLOT:
        return _parse_schema_regex(context)

    tables = []
    try:
        for stmt in sqlglot.parse(context, error_level=sqlglot.ErrorLevel.IGNORE):
            if stmt is None or not isinstance(stmt, exp.Create):
                continue
            table_expr = stmt.this
            if not isinstance(table_expr, exp.Schema):
                continue
            table_name = table_expr.this.name
            columns = []
            for col_def in table_expr.expressions:
                if isinstance(col_def, exp.ColumnDef):
                    col_name = col_def.name
                    col_type = col_def.kind
                    type_str = col_type.sql() if col_type else "TEXT"
                    columns.append({"name": col_name, "type": type_str})
            if columns:
                tables.append({"name": table_name, "columns": columns})
    except Exception:
        return _parse_schema_regex(context)

    return tables if tables else _parse_schema_regex(context)


# ── SQLite DB construction ────────────────────────────────────────────────────

def build_sqlite_db(context: str, tables: list[dict], n_rows: int = 20, reference_sql: str = "") -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")

    # Try transpiling CREATE TABLE statements to SQLite dialect
    created = set()
    if HAS_SQLGLOT:
        try:
            transpiled = sqlglot.transpile(context, write="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
            for stmt in transpiled:
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                        # Extract table name from transpiled stmt to track what was created
                        m = re.search(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?"?(\w+)"?', stmt, re.IGNORECASE)
                        if m:
                            created.add(m.group(1).lower())
                    except sqlite3.Error:
                        pass
            conn.commit()
        except Exception:
            pass

    # Fallback: create any tables that weren't created above
    for table in tables:
        if table["name"].lower() in created:
            continue
        col_defs = ", ".join(f'"{c["name"]}" TEXT' for c in table["columns"])
        try:
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table["name"]}" ({col_defs})')
        except sqlite3.Error:
            pass
    conn.commit()

    # Extract WHERE literals from reference SQL to inject into row 0
    where_values = extract_where_values(reference_sql) if reference_sql else {}

    # Insert synthetic rows
    for table in tables:
        col_names = ", ".join(f'"{c["name"]}"' for c in table["columns"])
        placeholders = ", ".join("?" * len(table["columns"]))
        type_cats = [_infer_type_category(c["type"]) for c in table["columns"]]

        rows = []
        for i in range(n_rows):
            row = []
            for j, col in enumerate(table["columns"]):
                # Row 0: inject WHERE values so reference SQL returns at least one row
                if i == 0 and col["name"].lower() in where_values:
                    row.append(where_values[col["name"].lower()])
                else:
                    row.append(_pool_value(type_cats[j], i, j))
            rows.append(tuple(row))

        try:
            conn.executemany(
                f'INSERT INTO "{table["name"]}" ({col_names}) VALUES ({placeholders})',
                rows,
            )
        except sqlite3.Error:
            pass
    conn.commit()
    return conn


# ── SQL execution & comparison ────────────────────────────────────────────────

def execute_sql(conn: sqlite3.Connection, sql: str) -> Optional[list]:
    sql = sql.strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        return None
    try:
        cursor = conn.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error:
        return None


def _normalize(rows: list) -> tuple:
    normalized = []
    for row in rows:
        normalized.append(tuple(
            str(v).strip().lower() if v is not None else "" for v in row
        ))
    return tuple(sorted(normalized))


def results_match(ref: list, pred: list) -> bool:
    return _normalize(ref) == _normalize(pred)


# ── Modal API call ────────────────────────────────────────────────────────────

def _strip_markdown(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?\s*", "", sql.strip(), flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql.strip())
    return sql.strip()


def call_modal_api(url: str, question: str, schema: str, timeout: int = 60) -> Optional[str]:
    try:
        resp = requests.post(
            url,
            json={"question": question, "schema": schema},
            timeout=timeout,
        )
        resp.raise_for_status()
        sql = resp.json().get("sql", "").strip()
        return _strip_markdown(sql) if sql else None
    except Exception:
        return None


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_exec_eval(modal_url: str, dataset, n_samples: int = 50, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    sample = dataset.select(indices.tolist())

    total = 0
    correct = 0
    skipped = 0

    print(f"Evaluating {len(sample)} samples via Modal...")
    print("First call may take 1-2 min (cold start)\n")

    for i, row in enumerate(sample):
        question = row["question"]
        schema = row["context"]
        reference_sql = row["answer"].strip()

        tables = parse_schema(schema)
        if not tables:
            skipped += 1
            continue

        conn = build_sqlite_db(schema, tables, reference_sql=reference_sql)

        ref_results = execute_sql(conn, reference_sql)
        if ref_results is None:
            skipped += 1
            conn.close()
            continue

        timeout = 180 if i == 0 else 45
        pred_sql = call_modal_api(modal_url, question, schema, timeout=timeout)

        if pred_sql is None:
            total += 1
            conn.close()
            continue

        pred_results = execute_sql(conn, pred_sql)
        conn.close()

        total += 1
        if pred_results is not None and results_match(ref_results, pred_results):
            correct += 1

        if (i + 1) % 10 == 0:
            acc = correct / total if total > 0 else 0.0
            print(f"  [{i+1:>3}/{len(sample)}]  acc: {acc:.1%}  ({correct}/{total})")

    execution_accuracy = round(correct / total, 4) if total > 0 else 0.0

    return {
        "execution_accuracy": execution_accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "n_samples_requested": n_samples,
    }
