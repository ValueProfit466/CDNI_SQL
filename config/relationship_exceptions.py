"""
Relationship Exceptions Registry

Explicit mappings for FK relationships where naming conventions fail.
These exceptions were identified from SPE_CDNI_Table_Relationship_Mapping_Guide.md
and represent the ~10% of cases where the standard pattern doesn't work.

Standard Pattern: ColumnNameId → columnname table
Exception Cases: Column name doesn't match table name or is context-dependent
"""

from typing import Dict, List, Optional

# Relationship exceptions: column name → table-specific mappings
RELATIONSHIP_EXCEPTIONS = {
    'RecipientId': {
        'tables': {
            'gostransaction': 'vesselecoidentifier',
            'transaction': 'vesselecoidentifier'
        },
        'reason': 'Recipient is vessel-eco identifier, not a "recipient" table',
        'confidence': 'HIGH',
        'source': 'SPE_CDNI_Table_Relationship_Mapping_Guide.md line 96'
    },
    'UserId': {
        'tables': {
            'gostransaction': 'appuser',
            'ecoidentifieruser': 'appuser',
            'audit': 'appuser',
            '*': 'appuser'  # Default for any table
        },
        'reason': 'Generic user reference - context determines meaning but typically appuser',
        'confidence': 'HIGH',
        'source': 'SPE_CDNI_Table_Relationship_Mapping_Guide.md line 27'
    },
    'FacilityUserId': {
        'tables': {
            'gostransaction': 'appuser',
            'wasteregistration': 'appuser',
            '*': 'appuser'  # Facility users are specialized app users
        },
        'reason': 'Facility user is specialized type of app user',
        'confidence': 'HIGH',
        'source': 'SPE_CDNI_Table_Relationship_Mapping_Guide.md line 28'
    },
    'ModifiedByUserId': {
        'tables': {
            '*': 'appuser'  # All tables with this audit column
        },
        'reason': 'Audit trail column - always references appuser',
        'confidence': 'HIGH',
        'source': 'SPE_CDNI_Table_Relationship_Mapping_Guide.md line 26'
    },
    'CreatedByUserId': {
        'tables': {
            '*': 'appuser'
        },
        'reason': 'Audit trail column - always references appuser',
        'confidence': 'HIGH',
        'source': 'Standard audit pattern'
    },
    'GosFacilityId': {
        'tables': {
            'gostransaction': 'gosfacility',
            '*': 'gosfacility'
        },
        'reason': 'Convention works but explicitly documented for clarity',
        'confidence': 'HIGH',
        'source': 'Domain knowledge'
    }
}

# Quick lookup: column name → target table (when unambiguous)
# Use this for columns that ALWAYS reference the same table regardless of source table
COLUMN_TARGET_OVERRIDES = {
    'RecipientId': 'vesselecoidentifier',
    'ModifiedByUserId': 'appuser',
    'CreatedByUserId': 'appuser',
}


def get_exception_target(column_name: str, source_table: str = None) -> Optional[str]:
    """
    Get the target table for an exception column.

    Args:
        column_name: The FK column name (e.g., 'RecipientId')
        source_table: The table containing this column (optional, for context-dependent lookups)

    Returns:
        Target table name or None if not an exception

    Examples:
        >>> get_exception_target('RecipientId')
        'vesselecoidentifier'

        >>> get_exception_target('UserId', 'gostransaction')
        'appuser'

        >>> get_exception_target('VesselId')
        None
    """
    # First check simple overrides
    if column_name in COLUMN_TARGET_OVERRIDES:
        return COLUMN_TARGET_OVERRIDES[column_name]

    # Then check context-dependent exceptions
    if column_name in RELATIONSHIP_EXCEPTIONS:
        exception = RELATIONSHIP_EXCEPTIONS[column_name]
        tables_map = exception['tables']

        # If source table specified, look for exact match
        if source_table and source_table in tables_map:
            return tables_map[source_table]

        # Fall back to wildcard if exists
        if '*' in tables_map:
            return tables_map['*']

    return None


def is_exception_column(column_name: str) -> bool:
    """
    Check if a column is a known exception.

    Args:
        column_name: The column name to check

    Returns:
        True if this column is in the exception registry
    """
    return (column_name in COLUMN_TARGET_OVERRIDES or
            column_name in RELATIONSHIP_EXCEPTIONS)


def get_all_exception_columns() -> List[str]:
    """
    Get list of all column names that are exceptions.

    Returns:
        List of column names that have exception mappings
    """
    all_columns = set(COLUMN_TARGET_OVERRIDES.keys())
    all_columns.update(RELATIONSHIP_EXCEPTIONS.keys())
    return sorted(list(all_columns))


def get_exception_info(column_name: str) -> Optional[Dict]:
    """
    Get detailed information about an exception.

    Args:
        column_name: The exception column name

    Returns:
        Dictionary with exception details or None if not an exception
    """
    if column_name in RELATIONSHIP_EXCEPTIONS:
        return RELATIONSHIP_EXCEPTIONS[column_name].copy()

    if column_name in COLUMN_TARGET_OVERRIDES:
        return {
            'tables': {'*': COLUMN_TARGET_OVERRIDES[column_name]},
            'reason': 'Unambiguous override',
            'confidence': 'HIGH',
            'source': 'Column target overrides'
        }

    return None


# Pre-defined junction table metadata
# These tables implement many-to-many relationships
JUNCTION_TABLES = {
    'vesselecoidentifier': {
        'type': 'many-to-many',
        'links': ['vessel', 'ecoidentifier'],
        'temporal': {
            'valid_from': 'ValidFromDateTime',
            'valid_until': 'ValidUntilDateTime',
            'null_means_active': True,
            'description': 'Tracks vessel-eco assignments over time'
        },
        'description': 'Links vessels to their ECO identifiers with temporal validity'
    },
    'vesselaccountholder': {
        'type': 'many-to-many',
        'links': ['vessel', 'accountholder'],
        'temporal': {
            'valid_from': 'ValidFrom',
            'valid_until': 'ValidUntil',
            'null_means_active': True,
            'description': 'Tracks vessel ownership/operation over time'
        },
        'description': 'Links vessels to account holders (owners/operators)'
    }
}


def is_junction_table(table_name: str) -> bool:
    """
    Check if a table is a known junction table.

    Args:
        table_name: The table name to check

    Returns:
        True if this is a junction table
    """
    return table_name in JUNCTION_TABLES


def get_junction_table_info(table_name: str) -> Optional[Dict]:
    """
    Get information about a junction table.

    Args:
        table_name: The junction table name

    Returns:
        Dictionary with junction table details or None
    """
    return JUNCTION_TABLES.get(table_name)
