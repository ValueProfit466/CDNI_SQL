"""
Data Quality Analyzer

Analyzes data quality metrics that affect relationship discovery:
- NULL rates per column
- Orphan records (FK values not found in target table)
- Cardinality analysis (1-to-1, 1-to-many, many-to-many)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyze data quality metrics for relationship profiling"""

    def __init__(self, data_directory: str, sample_rows: Optional[int] = None, interactive: bool = False):
        """
        Initialize analyzer with data directory.

        Args:
            data_directory: Path to directory containing CSV files
            sample_rows: Optional row cap to speed up profiling
            interactive: If True, print guidance when CSVs are missing
        """
        self.data_dir = Path(data_directory)
        self.csv_separator = ';'
        self.csv_encoding = 'utf-8-sig'
        self.sample_rows = sample_rows
        self.interactive = interactive
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def load_csv(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load CSV file for a table (cached, optional sampling)."""
        if table_name in self._cache:
            return self._cache[table_name]

        patterns = [
            f"{table_name}.csv",
            f"{table_name}_sample.csv",
            f"{table_name.lower()}.csv",
            f"{table_name.lower()}_sample.csv"
        ]

        df = None
        found_path = None
        for pattern in patterns:
            csv_path = self.data_dir / pattern
            if csv_path.exists():
                found_path = csv_path
                try:
                    df = pd.read_csv(
                        csv_path,
                        sep=self.csv_separator,
                        encoding=self.csv_encoding,
                        low_memory=False,
                        nrows=self.sample_rows
                    )
                    break
                except Exception as e:
                    logger.error(f"Error loading {csv_path}: {e}")
                    df = None
                    break

        if df is None and self.interactive:
            logger.warning(
                f"No CSV found for table '{table_name}'. "
                f"Run and export: SELECT * FROM {table_name} LIMIT 10000; "
                f"save as {self.data_dir}/{table_name}.csv"
            )

        self._cache[table_name] = df
        return df

    def analyze_null_rates(self, table_name: str) -> Dict[str, float]:
        """
        Calculate NULL rate for each column.

        Args:
            table_name: Table name

        Returns:
            Dictionary mapping column names to NULL rates (0.0 to 1.0)
            Example: {'CountryId': 0.40, 'VesselTypeId': 0.02, ...}
        """
        df = self.load_csv(table_name)
        if df is None:
            logger.warning(f"Could not load table: {table_name}")
            return {}

        null_rates = {}
        for column in df.columns:
            null_count = df[column].isna().sum()
            total_count = len(df)
            null_rates[column] = null_count / total_count if total_count > 0 else 0.0

        return null_rates

    def find_orphan_records(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str = 'Id'
    ) -> Dict:
        """
        Find FK values that don't exist in target table.

        Args:
            source_table: Source table name
            source_column: FK column in source table
            target_table: Target table name
            target_column: PK column in target table (default: 'Id')

        Returns:
            Dictionary with orphan analysis:
            {
                'orphan_count': 150,
                'orphan_percentage': 0.03,
                'sample_orphan_ids': [12345, 67890, ...],
                'impact': 'LOW'  # LOW/MEDIUM/HIGH based on percentage
            }
        """
        source_df = self.load_csv(source_table)
        target_df = self.load_csv(target_table)

        if source_df is None or target_df is None:
            return {
                'orphan_count': 0,
                'orphan_percentage': 0.0,
                'sample_orphan_ids': [],
                'impact': 'UNKNOWN',
                'error': 'Could not load tables'
            }

        if source_column not in source_df.columns:
            return {
                'orphan_count': 0,
                'orphan_percentage': 0.0,
                'sample_orphan_ids': [],
                'impact': 'UNKNOWN',
                'error': f'Column {source_column} not found in {source_table}'
            }

        if target_column not in target_df.columns:
            return {
                'orphan_count': 0,
                'orphan_percentage': 0.0,
                'sample_orphan_ids': [],
                'impact': 'UNKNOWN',
                'error': f'Column {target_column} not found in {target_table}'
            }

        # Get unique non-null FK values
        source_values = set(source_df[source_column].dropna().unique())
        target_values = set(target_df[target_column].dropna().unique())

        # Find orphans
        orphan_values = source_values.difference(target_values)
        orphan_count = len(orphan_values)
        orphan_percentage = orphan_count / len(source_values) if source_values else 0.0

        # Determine impact
        if orphan_percentage >= 0.10:
            impact = 'HIGH'
        elif orphan_percentage >= 0.05:
            impact = 'MEDIUM'
        else:
            impact = 'LOW'

        return {
            'orphan_count': orphan_count,
            'orphan_percentage': orphan_percentage,
            'sample_orphan_ids': sorted(list(orphan_values))[:10],  # First 10 orphans
            'total_source_values': len(source_values),
            'impact': impact,
            'recommendation': self._get_orphan_recommendation(impact, orphan_percentage)
        }

    def _get_orphan_recommendation(self, impact: str, percentage: float) -> str:
        """Get recommendation based on orphan impact."""
        if impact == 'HIGH':
            return (
                f"High orphan rate ({percentage:.1%}) - data quality issue detected. "
                "These records will be excluded from JOIN queries. Consider data cleanup."
            )
        elif impact == 'MEDIUM':
            return (
                f"Moderate orphan rate ({percentage:.1%}) - acceptable for optional relationships. "
                "Use LEFT JOIN to preserve these records if needed."
            )
        else:
            return (
                f"Low orphan rate ({percentage:.1%}) - expected for real-world data. "
                "No action needed."
            )

    def analyze_cardinality(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str = 'Id'
    ) -> Dict:
        """
        Determine relationship cardinality.

        Args:
            source_table: Source table name
            source_column: FK column in source table
            target_table: Target table name
            target_column: PK column in target table

        Returns:
            Dictionary with cardinality analysis:
            {
                'type': '1-to-many',  # or '1-to-1', 'many-to-many'
                'avg_children': 2.3,
                'max_children': 15,
                'min_children': 0,
                'distribution_summary': {...}
            }
        """
        source_df = self.load_csv(source_table)
        target_df = self.load_csv(target_table)

        if source_df is None or target_df is None:
            return {
                'type': 'UNKNOWN',
                'error': 'Could not load tables'
            }

        if source_column not in source_df.columns or target_column not in target_df.columns:
            return {
                'type': 'UNKNOWN',
                'error': 'Column not found'
            }

        # Count occurrences of each target value in source
        source_counts = source_df[source_column].value_counts()

        # Statistics
        avg_children = source_counts.mean() if len(source_counts) > 0 else 0
        max_children = source_counts.max() if len(source_counts) > 0 else 0
        min_children = source_counts.min() if len(source_counts) > 0 else 0

        # Determine relationship type
        if max_children == 1:
            rel_type = '1-to-1'
        elif avg_children < 1.5 and max_children > 1:
            rel_type = '1-to-1-or-many'
        else:
            rel_type = '1-to-many'

        # Check if it could be many-to-many (needs junction table analysis)
        # A junction table typically has FKs with high average children counts on both sides
        if avg_children > 2 and max_children > 10:
            rel_type = 'potentially-many-to-many'

        # Distribution summary
        distribution = {
            'single_child': (source_counts == 1).sum(),
            'multiple_children': (source_counts > 1).sum(),
            'total_target_values': len(source_counts)
        }

        return {
            'type': rel_type,
            'avg_children': float(avg_children),
            'max_children': int(max_children),
            'min_children': int(min_children),
            'distribution_summary': distribution,
            'interpretation': self._interpret_cardinality(rel_type, avg_children, max_children)
        }

    def _interpret_cardinality(self, rel_type: str, avg: float, max_val: int) -> str:
        """Generate human-readable interpretation of cardinality."""
        interpretations = {
            '1-to-1': f"Strict one-to-one relationship. Each {rel_type} value appears exactly once.",
            '1-to-1-or-many': f"Mostly one-to-one with some one-to-many (avg: {avg:.1f}, max: {max_val})",
            '1-to-many': f"One-to-many relationship. On average {avg:.1f} children per parent, up to {max_val}.",
            'potentially-many-to-many': f"High cardinality (avg: {avg:.1f}, max: {max_val}) - may be junction table."
        }
        return interpretations.get(rel_type, "Unknown relationship pattern")

    def analyze_table_completeness(self, table_name: str) -> Dict:
        """
        Analyze overall data completeness for a table.

        Returns:
            {
                'row_count': 10000,
                'column_count': 25,
                'overall_null_rate': 0.15,
                'columns_with_high_nulls': ['Imo', 'Tonnage'],
                'is_sample': True,
                'completeness_score': 0.85  # 0-1, higher is better
            }
        """
        df = self.load_csv(table_name)
        if df is None:
            return {'error': f'Could not load table: {table_name}'}

        null_rates = self.analyze_null_rates(table_name)
        avg_null_rate = sum(null_rates.values()) / len(null_rates) if null_rates else 0.0

        high_null_columns = [col for col, rate in null_rates.items() if rate > 0.50]

        # Check if this is sample data
        is_sample = '_sample' in str(self.data_dir / f"{table_name}.csv")

        # Completeness score: inverse of average null rate
        completeness_score = 1.0 - avg_null_rate

        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'overall_null_rate': avg_null_rate,
            'columns_with_high_nulls': high_null_columns,
            'high_null_count': len(high_null_columns),
            'is_sample': is_sample,
            'completeness_score': completeness_score,
            'data_quality_grade': self._grade_completeness(completeness_score)
        }

    def _grade_completeness(self, score: float) -> str:
        """Convert completeness score to letter grade."""
        if score >= 0.95:
            return 'A'
        elif score >= 0.85:
            return 'B'
        elif score >= 0.70:
            return 'C'
        elif score >= 0.50:
            return 'D'
        else:
            return 'F'

    def generate_quality_report(self, table_name: str) -> str:
        """
        Generate comprehensive data quality report for a table.

        Args:
            table_name: Table name

        Returns:
            Formatted string report
        """
        completeness = self.analyze_table_completeness(table_name)
        null_rates = self.analyze_null_rates(table_name)

        report = []
        report.append(f"\n{'='*60}")
        report.append(f"DATA QUALITY REPORT: {table_name}")
        report.append(f"{'='*60}")

        if 'error' in completeness:
            report.append(f"ERROR: {completeness['error']}")
            return '\n'.join(report)

        report.append(f"\nOverall Statistics:")
        report.append(f"  Rows: {completeness['row_count']:,}")
        report.append(f"  Columns: {completeness['column_count']}")
        report.append(f"  Sample Data: {'Yes' if completeness['is_sample'] else 'No'}")
        report.append(f"  Completeness Score: {completeness['completeness_score']:.1%} (Grade: {completeness['data_quality_grade']})")
        report.append(f"  Average NULL Rate: {completeness['overall_null_rate']:.1%}")

        if completeness['high_null_count'] > 0:
            report.append(f"\nColumns with High NULL Rates (>50%):")
            for col in completeness['columns_with_high_nulls']:
                rate = null_rates.get(col, 0.0)
                report.append(f"  - {col}: {rate:.1%}")

        report.append(f"\n{'='*60}\n")
        return '\n'.join(report)
