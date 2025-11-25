"""
Schema Navigator
Wraps the existing schema_explorer.py to provide query-builder-specific functionality.
Enhanced with hybrid FK discovery using relationship metadata.
"""

from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import sys
import json

# Import the existing SchemaExplorer
from schema_explorer import SchemaExplorer

# Import relationship exceptions
try:
    from config.relationship_exceptions import (
        get_exception_target,
        is_exception_column,
        JUNCTION_TABLES
    )
    HAS_EXCEPTIONS = True
except ImportError:
    HAS_EXCEPTIONS = False


class SchemaNavigator:
    """
    Wrapper around SchemaExplorer that provides query-builder-specific methods.
    Does NOT modify the original schema_explorer.py.
    Enhanced with hybrid relationship discovery from metadata.
    """

    def __init__(self, excel_path: str, metadata_path: str = None):
        """
        Initialize with path to schema Excel file.

        Args:
            excel_path: Path to the schema Excel file
            metadata_path: Optional path to relationship_metadata.json (for enhanced discovery)
        """
        self.explorer = SchemaExplorer(excel_path, enable_file_output=False)
        self.tables = self.explorer.tables
        self.relationships = self.explorer.relationships
        self.df = self.explorer.df
        self.enriched_tables = set()

        # Simple in-memory caches to avoid recomputing metadata-heavy lookups
        self._table_info_cache: Dict[str, Dict] = {}
        self._join_path_cache: Dict[Tuple[str, str, int], List[List[str]]] = {}

        # Load enhanced metadata if available
        self.metadata = self._load_metadata(metadata_path)
        self.has_enhanced_metadata = bool(self.metadata.get('relationships'))

    def _load_metadata(self, path: str) -> Dict:
        """Load pre-computed relationship metadata."""
        if not path:
            # Try default path
            default_path = Path('config/relationship_metadata.json')
            if default_path.exists():
                path = str(default_path)
            else:
                return {'relationships': {}, 'metadata': {}}

        path = Path(path)
        if path.exists():
            try:
                with open(path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata from {path}: {e}")
                return {'relationships': {}, 'metadata': {}}

        return {'relationships': {}, 'metadata': {}}

    def find_join_path(self, start_table: str, end_table: str, max_depth: int = 3) -> List[List[str]]:
        """
        Find possible join paths between two tables, sorted by length.

        Args:
            start_table: Starting table name
            end_table: Ending table name
            max_depth: Maximum join depth

        Returns:
            List of paths, each path is a list of table names
        """
        cache_key = (start_table, end_table, max_depth)
        if cache_key not in self._join_path_cache:
            paths = self.explorer.find_path(start_table, end_table, max_depth)
            self._join_path_cache[cache_key] = sorted(paths, key=len)
        return self._join_path_cache[cache_key]

    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """
        Get comprehensive information about a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information or None if table doesn't exist
        """
        if table_name not in self.tables:
            return None

        if table_name not in self._table_info_cache:
            table_df = self.df[self.df['TABLE_NAME'] == table_name]
            self._table_info_cache[table_name] = {
                'name': table_name,
                'columns': self._get_columns(table_name),
                'foreign_keys': self.relationships.get(table_name, []),
                'incoming_refs': self.explorer._find_incoming_relationships(table_name),
                'column_count': len(table_df)
            }

        return self._table_info_cache[table_name]

    def _get_columns(self, table_name: str) -> List[Dict]:
        """Get list of columns for a table with their metadata"""
        table_df = self.df[self.df['TABLE_NAME'] == table_name]
        columns = []

        for _, row in table_df.iterrows():
            col_info = {
                'name': row['COLUMN_NAME'],
                'data_type': row['DATA_TYPE'],
                'is_primary_key': row['COLUMN_NAME'] == 'Id',
                'is_foreign_key': row['COLUMN_NAME'].endswith('Id') and row['COLUMN_NAME'] != 'Id'
            }
            columns.append(col_info)

        return columns

    def get_foreign_keys(self, table: str) -> List[Tuple[str, str, str]]:
        """
        Get foreign key relationships for a table.

        Args:
            table: Table name

        Returns:
            List of tuples: (fk_column, target_table, data_type)
        """
        return self.relationships.get(table, [])

    def get_foreign_keys_enhanced(self, table: str) -> List[Tuple[str, str, str, Dict]]:
        """
        Get FK relationships using hybrid three-tier discovery.

        Three-tier approach:
        1. Check exceptions first (hardcoded mappings)
        2. Check value-based metadata (from profiling)
        3. Fall back to convention-based (naming patterns)

        Args:
            table: Table name

        Returns:
            List of tuples: (fk_column, target_table, data_type, metadata)

            metadata includes:
            - discovery_method: 'exception' | 'value_based' | 'convention'
            - confidence: 'HIGH' | 'MEDIUM' | 'LOW'
            - match_rate: 0.0-1.0 (if available)
            - is_sampled: bool (if using sample data)
            - warnings: list of strings
        """
        enhanced_fks = []

        # Get columns for this table
        table_df = self.df[self.df['TABLE_NAME'] == table]

        for _, row in table_df.iterrows():
            col = row['COLUMN_NAME']
            dtype = row['DATA_TYPE']

            # Only process FK candidates (ends with 'Id', not primary key)
            if not (col.endswith('Id') and col != 'Id'):
                continue

            # TIER 1: Check exceptions first
            if HAS_EXCEPTIONS:
                target = get_exception_target(col, table)
                if target:
                    enhanced_fks.append((col, target, dtype, {
                        'discovery_method': 'exception',
                        'confidence': 'HIGH',
                        'match_rate': 1.0,
                        'is_sampled': False,
                        'verified': True,
                        'source': 'relationship_exceptions.py'
                    }))
                    continue

            # TIER 2: Check value-based metadata
            if self.has_enhanced_metadata:
                if table in self.metadata.get('relationships', {}):
                    if col in self.metadata['relationships'][table]:
                        rel_meta = self.metadata['relationships'][table][col].copy()
                        enhanced_fks.append((
                            col,
                            rel_meta['target_table'],
                            dtype,
                            rel_meta
                        ))
                        continue

            # TIER 3: Try convention-based
            target = col[:-2].lower()  # Remove 'Id' suffix
            if target in self.tables:
                enhanced_fks.append((col, target, dtype, {
                    'discovery_method': 'convention',
                    'confidence': 'MEDIUM',
                    'match_rate': None,  # Not validated
                    'is_sampled': False,
                    'verified': False,
                    'warnings': ['Convention-based inference, not validated with data']
                }))

        return enhanced_fks

    def get_relationship_confidence(self, source_table: str, fk_column: str) -> Optional[Dict]:
        """
        Get confidence information for a specific FK relationship.

        Args:
            source_table: Source table name
            fk_column: FK column name

        Returns:
            Dictionary with confidence info or None
        """
        if self.has_enhanced_metadata:
            if source_table in self.metadata.get('relationships', {}):
                if fk_column in self.metadata['relationships'][source_table]:
                    return self.metadata['relationships'][source_table][fk_column]

        return None

    def is_junction_table(self, table_name: str) -> bool:
        """
        Check if a table is a known junction table.

        Args:
            table_name: Table name

        Returns:
            True if this is a junction table
        """
        if HAS_EXCEPTIONS:
            from config.relationship_exceptions import is_junction_table
            return is_junction_table(table_name)

        # Fallback: check metadata
        if 'junction_tables' in self.metadata:
            return table_name in self.metadata['junction_tables']

        return False

    def get_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """
        Determine the join condition between two tables.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            Join condition string or None if no direct relationship
        """
        # Check if table1 has FK to table2
        if table1 in self.relationships:
            for fk_col, target, _ in self.relationships[table1]:
                if target == table2:
                    return f"{table1}.{fk_col} = {table2}.Id"

        # Check if table2 has FK to table1
        if table2 in self.relationships:
            for fk_col, target, _ in self.relationships[table2]:
                if target == table1:
                    return f"{table2}.{fk_col} = {table1}.Id"

        return None

    def _score_columns(self, table_df) -> Dict[str, Dict]:
        """
        Score columns for filter/aggregation/grouping suitability.
        Returns a dict keyed by column with attributes.
        """
        scores: Dict[str, Dict] = {}
        for _, row in table_df.iterrows():
            col = row['COLUMN_NAME']
            dtype = row['DATA_TYPE']
            scores[col] = {
                'data_type': dtype,
                'is_status': self._is_status_column(col),
                'is_date': self._is_date_column(col),
                'is_fk': self._is_fk(col),
                'is_lookup_fk': self._is_lookup_fk(col),
                'is_numeric': self._is_numeric_column(col, dtype),
                'is_categorical': self._is_categorical_column(col, dtype),
                'semantic': 'amount' if any(word in col.lower() for word in ['amount', 'quantity', 'volume', 'total', 'price']) else None
            }
        return scores

    def suggest_filters_for_table(self, table_name: str) -> List[Dict]:
        """Suggest likely filter columns based on scored attributes."""
        table_df = self.df[self.df['TABLE_NAME'] == table_name]
        scores = self._score_columns(table_df)
        filter_candidates = []

        for col, meta in scores.items():
            if meta['is_status']:
                filter_candidates.append({
                    'column': col,
                    'type': 'status',
                    'data_type': meta['data_type'],
                    'description': f'Filter by {col}'
                })
            elif meta['is_date']:
                filter_candidates.append({
                    'column': col,
                    'type': 'date_range',
                    'data_type': meta['data_type'],
                    'description': f'Filter by date range on {col}'
                })
            elif meta['is_lookup_fk']:
                target_table = col[:-2].lower()
                filter_candidates.append({
                    'column': col,
                    'type': 'lookup',
                    'data_type': meta['data_type'],
                    'reference_table': target_table,
                    'description': f'Filter by {target_table}'
                })

        return filter_candidates

    def suggest_aggregation_columns(self, table_name: str) -> List[Dict]:
        """Suggest columns suitable for aggregation (SUM, AVG, COUNT)."""
        table_df = self.df[self.df['TABLE_NAME'] == table_name]
        scores = self._score_columns(table_df)
        agg_candidates = []

        for col, meta in scores.items():
            if meta['is_numeric']:
                if meta['semantic'] == 'amount':
                    agg_candidates.append({
                        'column': col,
                        'data_type': meta['data_type'],
                        'suggested_agg': ['SUM', 'AVG'],
                        'description': f'Sum or average of {col}'
                    })
                else:
                    agg_candidates.append({
                        'column': col,
                        'data_type': meta['data_type'],
                        'suggested_agg': ['COUNT', 'AVG'],
                        'description': f'Count or average of {col}'
                    })

        # Always suggest COUNT on primary key
        agg_candidates.append({
            'column': 'Id',
            'data_type': 'int',
            'suggested_agg': ['COUNT'],
            'description': f'Count of {table_name} records'
        })

        return agg_candidates

    def suggest_grouping_columns(self, table_name: str) -> List[Dict]:
        """Suggest columns suitable for GROUP BY."""
        table_df = self.df[self.df['TABLE_NAME'] == table_name]
        scores = self._score_columns(table_df)
        group_candidates = []

        for col, meta in scores.items():
            if meta['is_date']:
                group_candidates.append({
                    'column': col,
                    'data_type': meta['data_type'],
                    'grouping_options': ['YEAR', 'MONTH', 'QUARTER'],
                    'description': f'Group by time period from {col}'
                })
            elif meta['is_fk'] and col[:-2].lower() in self.tables:
                target_table = col[:-2].lower()
                group_candidates.append({
                    'column': col,
                    'data_type': meta['data_type'],
                    'reference_table': target_table,
                    'description': f'Group by {target_table}'
                })
            elif meta['is_categorical']:
                group_candidates.append({
                    'column': col,
                    'data_type': meta['data_type'],
                    'description': f'Group by {col}'
                })

        return group_candidates

    def build_join_chain(self, path: List[str]) -> List[Dict]:
        """
        Build detailed join information for a path.

        Args:
            path: List of table names representing the join path

        Returns:
            List of join dictionaries with conditions
        """
        joins = []

        for i in range(len(path) - 1):
            current = path[i]
            next_table = path[i + 1]

            join_condition = self.get_join_condition(current, next_table)

            if join_condition:
                joins.append({
                    'from_table': current,
                    'to_table': next_table,
                    'condition': join_condition,
                    'type': 'INNER JOIN'  # Default, can be changed later
                })

        return joins

    # ---------- Column classification helpers (internal) ----------
    def _is_status_column(self, col: str) -> bool:
        return any(word in col.lower() for word in ['status', 'state', 'operational'])

    def _is_date_column(self, col: str) -> bool:
        return any(word in col.lower() for word in ['date', 'time', 'datetime'])

    def _is_fk(self, col: str) -> bool:
        return col.endswith('Id') and col != 'Id'

    def _is_lookup_fk(self, col: str) -> bool:
        if not self._is_fk(col):
            return False
        target_table = col[:-2].lower()
        return target_table in ['country', 'vesseltype', 'fueltype', 'wastetype', 'hulltype']

    def _is_numeric_column(self, col: str, dtype: str) -> bool:
        if self._is_fk(col):
            return False
        return any(num_type in dtype.lower() for num_type in ['int', 'decimal', 'float', 'numeric'])

    def _is_categorical_column(self, col: str, dtype: str) -> bool:
        return ('varchar' in dtype.lower() and any(word in col.lower() for word in ['name', 'code', 'type', 'status']))

    def find_tables_for_entities(self, entities: List[str], min_similarity_ratio: float = 0.5) -> Dict[str, List[str]]:
        """
        Map entity names to database tables with confidence tracking.

        Args:
            entities: List of entity names (e.g., ['vessel', 'transaction'])
            min_similarity_ratio: Minimum ratio of entity name that must match table name (0.0-1.0)

        Returns:
            Dictionary mapping entities to their primary and related tables

        Note:
            Also populates self._last_entity_confidence with confidence levels:
            - 'HIGH': Exact match from entity_table_map
            - 'MEDIUM': Enriched from relationship metadata
            - 'LOW': Similarity-based fallback (use with caution)
        """
        entity_table_map = {
            'vessel': ['vessel', 'vesselecoidentifier', 'vesselaccountholder'],
            'transaction': ['transaction', 'gostransaction', 'transactiontariff'],
            'waste': ['wasteregistration', 'wasteregistrationdetail', 'wastetype', 'wastecollectionfacility', 'wastecollection', 'wastecollectiondetail','wastetypelocale','wastecollectionfacilitylocale','wasteregistrationlocale','wastecollectioncompany', 'transactiondataimport', 'transactiontariff'],
            'account': ['account', 'accountholder', 'ecoidentifier', 'reqecoaccountvessels'],
            'country': ['country', 'nationalinstitute', 'countrylocale', 'translationlanguage'],
            'facility': ['gosfacility', 'gos', 'wastecollectionfacility','goscontactinformation'],
            'fuel': ['fueltype', 'fueltypelocale']
        }

        # Track confidence for each entity mapping
        self._last_entity_confidence: Dict[str, str] = {}

        result = {}
        for entity in entities:
            mapped = entity_table_map.get(entity, [])
            matched = [t for t in mapped if t in self.tables]
            if matched:
                result[entity] = matched
                self._last_entity_confidence[entity] = 'HIGH'
                continue

            # Enrichment: if relationships show targets linked to this entity name, include them
            enriched = self._find_enriched_tables_for_entity(entity)
            if enriched:
                result[entity] = enriched
                self._last_entity_confidence[entity] = 'MEDIUM'
                continue

            # Fallback: suggest similar tables with threshold check
            similar = self._find_similar_tables_with_threshold(entity, min_similarity_ratio)
            if similar:
                result[entity] = similar[:3]
                self._last_entity_confidence[entity] = 'LOW'
                print(f"Warning: '{entity}' not in known entities. Using similar tables: {similar[:3]} (LOW confidence)")

        return result

    def _find_similar_tables_with_threshold(self, partial_name: str, min_ratio: float = 0.5) -> List[str]:
        """
        Find tables with similar names, filtered by minimum match ratio.

        Args:
            partial_name: Partial table name to search for
            min_ratio: Minimum ratio of partial_name length that must match

        Returns:
            List of matching table names, sorted by match quality
        """
        partial_lower = partial_name.lower()
        min_match_len = int(len(partial_lower) * min_ratio)

        candidates = []
        for t in self.tables:
            t_lower = t.lower()
            # Check if partial matches table or table matches partial
            if partial_lower in t_lower:
                # Compute match quality: ratio of matched chars to table name length
                match_quality = len(partial_lower) / len(t_lower)
                candidates.append((t, match_quality))
            elif t_lower in partial_lower and len(t_lower) >= min_match_len:
                match_quality = len(t_lower) / len(partial_lower)
                candidates.append((t, match_quality))

        # Sort by match quality descending
        candidates.sort(key=lambda x: -x[1])
        return [t for t, _ in candidates[:10]]

    def get_entity_confidence(self, entity: str) -> Optional[str]:
        """
        Get the confidence level for a recent entity mapping.

        Returns:
            'HIGH', 'MEDIUM', 'LOW', or None if entity not recently mapped
        """
        return getattr(self, '_last_entity_confidence', {}).get(entity)

    def _find_enriched_tables_for_entity(self, entity: str) -> List[str]:
        """
        Use enriched relationship info to suggest tables for an entity name.
        Currently matches entity name to table names that appear as targets in relationships.
        """
        entity_lower = entity.lower()
        targets = set()
        for src, rels in self.relationships.items():
            for _, tgt, _ in rels:
                if entity_lower in tgt.lower():
                    if tgt in self.tables:
                        targets.add(tgt)
        return sorted(list(targets))[:3]

    def find_similar_tables(self, partial_name: str) -> List[str]:
        """
        Find tables with similar names (fuzzy matching).

        Args:
            partial_name: Partial table name

        Returns:
            List of matching table names
        """
        partial_lower = partial_name.lower()
        similar = [t for t in self.tables if partial_lower in t.lower()]
        return sorted(similar)[:10]  # Return top 10 matches

    def validate_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the schema"""
        return table_name in self.tables

    def get_table_relationships_summary(self, table_name: str) -> Dict:
        """
        Get a summary of all relationships for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with outgoing and incoming relationships
        """
        if table_name not in self.tables:
            return {}

        outgoing = self.relationships.get(table_name, [])
        incoming = self.explorer._find_incoming_relationships(table_name)

        return {
            'table': table_name,
            'outgoing_relationships': [
                {'column': fk, 'references': target} for fk, target, _ in outgoing
            ],
            'incoming_relationships': [
                {'from_table': source, 'via_column': col} for source, col in incoming
            ],
            'total_relationships': len(outgoing) + len(incoming)
        }
