# README.md

## Project Overview

SPE-CDNI Query Builder is a Python CLI tool that helps users build SQL queries for the SPE-CDNI database (European inland shipping environmental compliance system) without requiring SQL knowledge. It parses natural language questions in English and Dutch to generate MySQL-compatible queries.

## Common Commands

```bash
# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Run the query builder CLI
python query_builder.py

# Run schema explorer (interactive)
python schema_explorer.py
python schema_explorer.py --enrich  # Enable template/CSV enrichment

# Non-interactive schema exploration
python schema_explorer.py explore vessel
python schema_explorer.py path vessel transaction
python schema_explorer.py list core

# Run tests
pytest tests/

# Install dependencies (if needed)
pip install pandas openpyxl
```

## Architecture

The query builder uses a pipeline architecture:

```
User Question → IntentInterpreter → SchemaNavigator → QueryGenerator → SQL Output
                  (NLP parsing)      (find joins)      (build SQL)
```

### Core Modules

- **[query_builder.py](query_builder.py)** - Main CLI entry point, orchestrates the pipeline, handles template matching
- **[schema_explorer.py](schema_explorer.py)** - Interactive schema exploration, relationship mapping, file output capture
- **[core/intent_interpreter.py](core/intent_interpreter.py)** - Keyword-based natural language parsing, entity detection, filter extraction
- **[core/schema_navigator.py](core/schema_navigator.py)** - Wraps SchemaExplorer for query-builder-specific operations, three-tier FK discovery
- **[generators/query_generator.py](generators/query_generator.py)** - SQL query construction with state tracking for joins, WHERE, GROUP BY

### Support Modules

- **[utils/data_catalog.py](utils/data_catalog.py)** - CSV-backed lookup values for filter suggestions (countries, waste types, facilities)
- **[config/relationship_exceptions.py](config/relationship_exceptions.py)** - Hardcoded FK mappings where naming conventions fail (e.g., `RecipientId` → `vesselecoidentifier`)

### Key Data Files

- **AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx** - Schema metadata source (required)
- **data/\*.csv** - Exported reference data from database (semicolon-delimited, UTF-8 BOM)
- **saved_queries/\*.sql** - Query templates for common patterns

## FK Discovery Strategy

The schema uses three-tier foreign key discovery:

1. **Exceptions** - Hardcoded mappings in `config/relationship_exceptions.py`
2. **Value-based** - Pre-computed metadata from `config/relationship_metadata.json`
3. **Convention** - Pattern: `ColumnNameId` → `columnname` table

Key FK exceptions to be aware of:

- `RecipientId` → `vesselecoidentifier` (not "recipient")
- `ModifiedByUserId`, `CreatedByUserId`, `UserId` → `appuser`
- `FacilityUserId` → `appuser`

## Entity-to-Table Mapping

| Entity      | Primary Tables                                        |
|-------------|-------------------------------------------------------|
| vessel      | vessel, vesselecoidentifier, vesselaccountholder      |
| transaction | transaction, gostransaction                           |
| waste       | wasteregistration, wasteregistrationdetail, wastetype |
| account     | account, accountholder, ecoidentifier                 |
| country     | country                                               |
| facility    | gosfacility, gos, wastecollectionfacility             |
| fuel        | fueltype                                              |

## Critical Join Patterns

Transaction to vessel (most common):

```sql
FROM transaction t
LEFT JOIN vesselecoidentifier vei ON vei.Id = t.RecipientId
LEFT JOIN vessel v ON v.Id = vei.VesselId
```

Vessel to account holder:

```sql
FROM vessel v
JOIN vesselaccountholder vah ON vah.VesselId = v.Id
JOIN accountholder ah ON ah.Id = vah.AccountHolderId
```

ECO account hierarchy:

```sql
FROM accountholder ah
JOIN account a ON a.AccountHolderId = ah.Id
JOIN ecoidentifier eco ON eco.AccountId = a.Id
```

## Transaction Status Codes

For `gostransaction.StatusCode`:

- `B` - Booked (processed, default filter)
- `I` - Initiated (pending)
- `P`, `W` - Pending/Withheld
- `F`, `D` - Failed/Declined

## Testing

Tests are in `tests/` using pytest:

- [tests/test_intent_interpreter.py](tests/test_intent_interpreter.py) - NLP parsing tests
- [tests/test_query_generator.py](tests/test_query_generator.py) - SQL generation tests
- [tests/test_schema_navigator.py](tests/test_schema_navigator.py) - Entity mapping and confidence tests

## Architecture Notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for:

- Component dependency diagram
- Tool workflow recommendations
- Known issues and risks
- Development guidelines

## CSV Data Notes

CSV exports in `data/` directory:

- Semicolon-delimited (`;`)
- UTF-8 with BOM encoding
- Match table names exactly (case-sensitive)
- Not committed to version control (contains production data)
