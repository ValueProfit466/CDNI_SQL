"""
Relationship Profiler

Discovers FK relationships by analyzing actual CSV data values.
Handles cases where naming conventions fail (e.g., RecipientId  vesselecoidentifier).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelationshipProfiler:
    """Discovers FK relationships by analyzing CSV data values"""

    def __init__(self, data_directory: str):
        """
        Initialize profiler with data directory.

        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_dir = Path(data_directory)
        self.csv_cache = {}  # Cache loaded CSVs: {table_name: DataFrame}
        self.csv_separator = ';'  # European CSV format
        self.csv_encoding = 'utf-8-sig'  # UTF-8 with BOM

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

    def load_csv(self, table_name: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load CSV file for a table.

        Args:
            table_name: Name of the table (without .csv extension)
            use_cache: Whether to use cached DataFrame if available

        Returns:
            DataFrame or None if file doesn't exist
        """
        # Check cache first
        if use_cache and table_name in self.csv_cache:
            logger.debug(f"Using cached data for {table_name}")
            return self.csv_cache[table_name]

        # Try different filename patterns
        patterns = [
            f"{table_name}.csv",
            f"{table_name}_sample.csv",
            f"{table_name.lower()}.csv",
            f"{table_name.lower()}_sample.csv"
        ]

        for pattern in patterns:
            csv_path = self.data_dir / pattern
            if csv_path.exists():
                try:
                    logger.info(f"Loading {csv_path.name}...")
                    df = pd.read_csv(
                        csv_path,
                        sep=self.csv_separator,
                        encoding=self.csv_encoding,
                        low_memory=False  # Prevent dtype warnings for large files
                    )
                    self.csv_cache[table_name] = df
                    logger.info(f"Loaded {len(df)} rows from {csv_path.name}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {csv_path}: {e}")
                    return None

        logger.warning(f"No CSV file found for table: {table_name}")
        return None

    def get_unique_values(self, df: pd.DataFrame, column: str) -> Set:
        """
        Get unique non-null values from a column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Set of unique non-null values
        """
        if column not in df.columns:
            return set()

        # Get non-null values and convert to set
        values = df[column].dropna().unique()
        return set(values)

    def profile_relationship(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_pk: str = 'Id'
    ) -> Dict:
        """
        Test if source_column values match target_table.Id values.

        Algorithm:
        1. Load source CSV, extract unique values from source_column
        2. Load target CSV, extract unique values from target_pk
        3. Calculate intersection (matched values)
        4. Return confidence metrics

        Args:
            source_table: Source table name
            source_column: Source column name (FK candidate)
            target_table: Target table name
            target_pk: Target primary key column (default: 'Id')

        Returns:
            Dictionary with profiling results:
            {
                'match_rate': 0.95,           # 95% of non-null values matched
                'sample_size': 10000,          # How many rows analyzed
                'matched_count': 9500,         # How many values found in target
                'null_count': 400,             # How many NULLs (excluded from match rate)
                'orphan_count': 500,           # Values NOT found in target
                'confidence': 'HIGH',          # HIGH (e95%), MEDIUM (80-95%), LOW (<80%)
                'is_sampled': False,           # Is this from sample data?
                'evidence': 'value_overlap',   # How we discovered this
                'warnings': []                 # Any warnings about data quality
            }
        """
        result = {
            'match_rate': 0.0,
            'sample_size': 0,
            'matched_count': 0,
            'null_count': 0,
            'orphan_count': 0,
            'confidence': 'NONE',
            'is_sampled': False,
            'evidence': 'value_overlap',
            'warnings': [],
            'target_table': target_table,
            'target_column': target_pk
        }

        # Load source data
        source_df = self.load_csv(source_table)
        if source_df is None:
            result['warnings'].append(f"Could not load source table: {source_table}")
            return result

        # Check if source column exists
        if source_column not in source_df.columns:
            result['warnings'].append(f"Column {source_column} not found in {source_table}")
            return result

        # Load target data
        target_df = self.load_csv(target_table)
        if target_df is None:
            result['warnings'].append(f"Could not load target table: {target_table}")
            return result

        # Check if target column exists
        if target_pk not in target_df.columns:
            result['warnings'].append(f"Column {target_pk} not found in {target_table}")
            return result

        # Check if this is sample data
        is_sampled = '_sample' in str(self.data_dir / f"{source_table}.csv") or \
                     '_sample' in str(self.data_dir / f"{target_table}.csv")
        result['is_sampled'] = is_sampled

        # Get unique values from source column
        source_values = self.get_unique_values(source_df, source_column)
        total_rows = len(source_df)
        null_count = source_df[source_column].isna().sum()
        non_null_count = total_rows - null_count

        result['sample_size'] = total_rows
        result['null_count'] = null_count

        if len(source_values) == 0:
            result['warnings'].append(f"No non-null values in {source_column}")
            return result

        # Get unique values from target column
        target_values = self.get_unique_values(target_df, target_pk)

        if len(target_values) == 0:
            result['warnings'].append(f"No non-null values in {target_table}.{target_pk}")
            return result

        # Calculate intersection
        matched_values = source_values.intersection(target_values)
        orphan_values = source_values.difference(target_values)

        result['matched_count'] = len(matched_values)
        result['orphan_count'] = len(orphan_values)

        # Calculate match rate (percentage of non-null values that matched)
        if len(source_values) > 0:
            result['match_rate'] = len(matched_values) / len(source_values)

        # Determine confidence level
        match_rate = result['match_rate']
        if match_rate >= 0.95:
            result['confidence'] = 'HIGH'
        elif match_rate >= 0.80:
            result['confidence'] = 'MEDIUM'
        elif match_rate >= 0.50:
            result['confidence'] = 'LOW'
        else:
            result['confidence'] = 'NONE'

        # Add warnings for data quality issues
        if is_sampled:
            result['warnings'].append(
                f"Using sample data - results may be incomplete"
            )

        if null_count > 0:
            null_pct = (null_count / total_rows) * 100
            result['null_rate'] = null_count / total_rows
            if null_pct > 50:
                result['warnings'].append(
                    f"High NULL rate: {null_pct:.1f}% of values are NULL"
                )

        if result['orphan_count'] > 0:
            orphan_pct = (result['orphan_count'] / len(source_values)) * 100
            if orphan_pct > 5:
                result['warnings'].append(
                    f"Orphan records: {result['orphan_count']} values "
                    f"({orphan_pct:.1f}%) not found in {target_table}"
                )

        return result

    def discover_all_relationships(self, table: str, candidate_tables: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Find all FK relationships for a table by testing every *Id column
        against candidate target tables.

        Args:
            table: Table name to analyze
            candidate_tables: List of candidate target tables (if None, tries all available CSVs)

        Returns:
            Dictionary mapping column names to list of potential relationships:
            {
                'VesselId': [{
                    'target_table': 'vessel',
                    'confidence': 'HIGH',
                    'match_rate': 0.99,
                    ...
                }],
                'RecipientId': [{
                    'target_table': 'vesselecoidentifier',
                    'confidence': 'HIGH',
                    'match_rate': 0.96,
                    ...
                }]
            }
        """
        logger.info(f"Discovering relationships for {table}...")

        # Load source table
        source_df = self.load_csv(table)
        if source_df is None:
            logger.error(f"Could not load table: {table}")
            return {}

        # Find all columns ending in 'Id' (excluding primary key 'Id')
        fk_candidates = [col for col in source_df.columns
                        if col.endswith('Id') and col != 'Id']

        logger.info(f"Found {len(fk_candidates)} FK candidates: {fk_candidates}")

        # Get list of candidate tables if not provided
        if candidate_tables is None:
            candidate_tables = self._get_available_tables()

        discoveries = {}

        for fk_column in fk_candidates:
            logger.info(f"Profiling {fk_column}...")
            column_discoveries = []

            # Try matching against each candidate table
            for target_table in candidate_tables:
                result = self.profile_relationship(table, fk_column, target_table)

                # Only include high or medium confidence matches
                if result['confidence'] in ['HIGH', 'MEDIUM']:
                    column_discoveries.append(result)

            # Sort by match rate (best first)
            column_discoveries.sort(key=lambda x: x['match_rate'], reverse=True)

            if column_discoveries:
                discoveries[fk_column] = column_discoveries
                logger.info(
                    f"  {fk_column}: Found {len(column_discoveries)} potential targets, "
                    f"best match: {column_discoveries[0]['target_table']} "
                    f"(match_rate: {column_discoveries[0]['match_rate']:.2%})"
                )
            else:
                logger.warning(f"  {fk_column}: No confident matches found")

        return discoveries

    def _get_available_tables(self) -> List[str]:
        """
        Get list of available table names from CSV files in data directory.

        Returns:
            List of table names (without .csv extension)
        """
        tables = []

        for csv_file in self.data_dir.glob('*.csv'):
            table_name = csv_file.stem  # Filename without extension
            # Remove '_sample' suffix if present
            table_name = table_name.replace('_sample', '')
            if table_name not in tables:
                tables.append(table_name)

        logger.debug(f"Found {len(tables)} tables in {self.data_dir}")
        return sorted(tables)

    def profile_table_pair(
        self,
        table1: str,
        table2: str,
        check_both_directions: bool = True
    ) -> Dict[str, Dict]:
        """
        Profile potential relationships between two tables in both directions.

        Args:
            table1: First table name
            table2: Second table name
            check_both_directions: Whether to check FK relationships both ways

        Returns:
            Dictionary with results for both directions:
            {
                'table1_to_table2': {...},
                'table2_to_table1': {...} (if check_both_directions=True)
            }
        """
        results = {}

        # Check table1  table2
        discoveries1 = self.discover_all_relationships(table1, [table2])
        if discoveries1:
            results['table1_to_table2'] = discoveries1

        # Check table2  table1
        if check_both_directions:
            discoveries2 = self.discover_all_relationships(table2, [table1])
            if discoveries2:
                results['table2_to_table1'] = discoveries2

        return results

    def clear_cache(self):
        """Clear the CSV data cache to free memory."""
        self.csv_cache.clear()
        logger.info("CSV cache cleared")

    def get_cache_size(self) -> int:
        """
        Get number of tables currently cached in memory.

        Returns:
            Number of cached tables
        """
        return len(self.csv_cache)
