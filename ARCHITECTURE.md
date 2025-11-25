# Architecture and Development Notes

This document captures the design decisions, known issues, and improvement roadmap for the SPE-CDNI Query Builder.

## Component Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  query_builder.py          schema_explorer.py        discovery/cli_*.py     │
│  (Main CLI)                (Schema exploration)      (Data profiling)       │
└────────────────┬───────────────────┬───────────────────────┬────────────────┘
                 │                   │                       │
┌────────────────▼───────────────────▼───────────────────────▼────────────────┐
│                              CORE MODULES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  core/intent_interpreter.py    core/schema_navigator.py                     │
│  (NLP parsing)                 (Relationship navigation)                    │
│                                                                             │
│  generators/query_generator.py  utils/data_catalog.py                       │
│  (SQL generation)               (CSV lookup values)                         │
└────────────────┬────────────────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Schema Excel file             config/relationship_metadata.json            │
│  (required)                    (optional - improves FK discovery)           │
│                                                                             │
│  data/*.csv                    saved_queries/*.sql                          │
│  (optional - filter values)    (optional - query templates)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tool Dependency Chain

### Recommended Setup Workflow

```text
1. Required: Schema Excel file must exist
       ↓
2. Optional: Export CSV reference data from database
       ↓
3. Optional: Run metadata builder to generate relationship_metadata.json
       ↓
4. Ready: Run query_builder.py or schema_explorer.py
```

### When to Run Each Tool

| Tool                                 | When to Run                             | Output                               |
| ------------------------------------ | --------------------------------------- | ------------------------------------ |
| python query_builder.py              | Main usage - building queries           | SQL queries to console               |
| python schema_explorer.py            | Exploring schema; finding relationships | Console + optional files             |
| python schema_explorer.py --enrich   | Enhanced mode with templates/CSVs       | Console + enriched output            |
| python -m discovery.metadata_builder | After adding new CSVs                   | config/relationship_metadata.json    |
| python -m discovery.eni_scanner      | One-time ENI analysis                   | ./discovery/cache/\*.csv             |
| python -m discovery.cli_profiler     | Profiling FK relationships              | docs/relationship_metadata_report.md |

## Known Issues and Risks

### 1. Schema Navigator Entity Mapping (MEDIUM RISK)

**Location**: `core/schema_navigator.py:find_tables_for_entities()`

**Issue**: The similarity fallback has no confidence thresholds. If entity mapping fails, it returns similar tables without warning users that these are guesses.

**Current behavior**:

```python
# Lines 491-494: Returns top 3 similar tables without scoring
similar = self.explorer.find_similar_tables(entity)
if similar:
    result[entity] = similar[:3]  # No threshold, no warning
```

**Risk**: Users may receive unexpected tables in queries without realizing they're fallback suggestions.

**Mitigation**: Add confidence scoring and warnings (see TODO section).

### 2. ENI Scanner Complexity (LOW RISK)

**Location**: `discovery/eni_scanner.py`

**Issue**: The scanner combines multiple concerns:

- ENI pattern detection
- Overlap analysis (which columns share values)
- Unique value listing

**Current behavior**: All stages run together, producing 3 output files.

**Risk**: Performance issues with large datasets; confusing output structure.

**Mitigation**: Separate into stages with clearer configuration (see TODO section).

### 3. Enrichment Opt-in (LOW RISK)

**Location**: `schema_explorer.py`

**Issue**: Enrichment features (template parsing, CSV awareness) are opt-in via `--enrich` flag. Users may not realize they exist.

**Mitigation**: Document in README and CLI help. Consider making enrichment default if dependencies exist.

### 4. Test Coverage Gaps (MEDIUM RISK)

**Current tests**:

- `tests/test_query_generator.py` - Basic SQL generation
- `tests/test_intent_interpreter.py` - NLP parsing

**Missing coverage**:

- Entity mapping fallback behavior
- ENI scanner logic
- Metadata builder edge cases
- Integration tests for full pipeline

## Output File Locations

### Runtime Cache (can be deleted)

| File                                      | Purpose                 |
| ----------------------------------------- | ----------------------- |
| `discovery/cache/eni_scanner_output.csv`  | ENI scan results        |
| `discovery/cache/eni_scanner_links.csv`   | Column overlap analysis |
| `discovery/cache/eni_scanner_uniques.csv` | Unique value listing    |

### Generated Documentation

| File                                   | Purpose            |
| -------------------------------------- | ------------------ |
| `docs/relationship_metadata_report.md` | FK analysis report |

### Configuration (persist)

| File                                | Purpose                  |
| ----------------------------------- | ------------------------ |
| `config/relationship_metadata.json` | Pre-computed FK mappings |
| `config/relationship_exceptions.py` | Hardcoded FK exceptions  |

### Data (do not commit)

| Directory    | Purpose                            |
| ------------ | ---------------------------------- |
| `data/*.csv` | Database exports for filter values |

## TODO: Architectural Improvements

### Priority 1: Navigator Confidence Scoring

Add confidence scoring to entity mapping:

```python
# Proposed change to find_tables_for_entities()
def find_tables_for_entities(self, entities: List[str]) -> Dict[str, Tuple[List[str], str]]:
    """
    Returns:
        Dict mapping entity to (tables, confidence)
        confidence: 'HIGH' (exact match), 'MEDIUM' (enriched), 'LOW' (similarity)
    """
```

Log warnings for LOW confidence mappings.

### Priority 2: ENI Scanner Refactor

Split into separate functions:

```python
def scan_eni_patterns(data_dir, ...) -> List[ENIResult]:
    """Stage 1: Detect ENI-like values"""
def analyze_overlaps(results, value_sets, ...) -> List[OverlapResult]:
    """Stage 2: Optional overlap analysis"""
def list_uniques(results, value_sets, ...) -> List[UniqueResult]:
    """Stage 3: Optional unique value listing"""
```

Add CLI flags to enable/disable stages independently.

### Priority 3: Test Coverage

Add tests for:

1. `test_navigator_fallback.py` - Entity mapping fallback behavior
2. `test_eni_scanner.py` - ENI pattern detection
3. `test_integration.py` - Full pipeline from question to SQL

### Priority 4: Consistent Output Format

Standardize all tool outputs:

- JSON for machine-readable data
- Markdown for human-readable reports
- CSV for tabular analysis

## Development Guidelines

### Adding New Entity Mappings

1. Add to `entity_table_map` in `schema_navigator.py`
2. Add keywords to `entity_keywords` in `intent_interpreter.py`
3. Test with `python query_builder.py` using the new entity

### Adding New FK Exceptions

1. Add to `RELATIONSHIP_EXCEPTIONS` in `config/relationship_exceptions.py`
2. Document the source/reason in the exception entry
3. Run metadata builder to regenerate `relationship_metadata.json`

### Debugging Query Generation

Enable verbose output:

```python
# In query_builder.py, after intent parsing:
print(f"Debug - Intent: {intent}")
print(f"Debug - Tables: {entity_tables}")
print(f"Debug - Join path: {join_path}")
```

## Version History

| Date    | Changes                                                      |
| ------- | ------------------------------------------------------------ |
| 2025-11 | Initial MVP: query_builder, schema_explorer, basic discovery |
| 2025-11 | Added enrichment, metadata builder, ENI scanner              |
| 2025-11 | Documentation reorganization (user_guide/, CLAUDE.md)        |
