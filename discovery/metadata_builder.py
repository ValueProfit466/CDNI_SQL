"""
Metadata Builder

Orchestrates profiling all CSVs and generates relationship_metadata.json.
Combines three sources of relationship data:
1. Hardcoded exceptions (relationship_exceptions.py)
2. Value-based profiling (relationship_profiler.py)
3. Convention-based inference (schema_explorer.py)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Add parent directory to path to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.relationship_profiler import RelationshipProfiler
from discovery.data_quality_analyzer import DataQualityAnalyzer
from schema_explorer import SchemaExplorer

# Import exceptions
from config.relationship_exceptions import (
    RELATIONSHIP_EXCEPTIONS,
    COLUMN_TARGET_OVERRIDES,
    JUNCTION_TABLES,
    get_exception_target
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class MetadataBuilder:
    """Builds and maintains relationship_metadata.json"""

    def __init__(
        self,
        data_dir: str,
        excel_schema: str,
        sample_rows: Optional[int] = None,
        enable_enrichment: bool = False,
        interactive: bool = False
    ):
        """
        Initialize metadata builder.

        Args:
            data_dir: Path to directory containing CSV files
            excel_schema: Path to schema Excel file
            sample_rows: Optional row cap for profiling
            enable_enrichment: If True, use template/CSV enrichment in SchemaExplorer
        """
        self.profiler = RelationshipProfiler(data_dir)
        self.quality_analyzer = DataQualityAnalyzer(data_dir, sample_rows=sample_rows, interactive=interactive)
        self.schema_explorer = SchemaExplorer(
            excel_schema,
            enable_file_output=False,
            enable_enrichment=enable_enrichment,
            data_dir=data_dir,
            template_dir="saved_queries"
        )
        self.data_dir = Path(data_dir)
        self.excel_schema = excel_schema

    def build_metadata(self, force_refresh: bool = False,
                      tables_to_profile: List[str] = None) -> Dict:
        """
        Build complete relationship metadata by combining:
        1. Convention-based relationships (from schema_explorer)
        2. Value-based discoveries (from profiler)
        3. Hardcoded exceptions (from relationship_exceptions.py)

        Args:
            force_refresh: If True, re-profile all tables even if metadata exists
            tables_to_profile: If provided, only profile these tables

        Returns:
            Complete metadata dictionary
        """
        logger.info("="*60)
        logger.info("BUILDING RELATIONSHIP METADATA")
        logger.info("="*60)

        metadata = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'csv_files_analyzed': 0,
                'total_relationships': 0,
                'version': '1.0',
                'data_directory': str(self.data_dir),
                'schema_file': str(self.excel_schema)
            },
            'relationships': {},
            'junction_tables': JUNCTION_TABLES.copy(),
            'data_quality': {},
            'exceptions_applied': {}
        }

        # Get list of tables to analyze
        if tables_to_profile is None:
            tables_to_profile = sorted(self.schema_explorer.tables)

        logger.info(f"Profiling {len(tables_to_profile)} tables...")

        relationship_count = 0

        for table in tables_to_profile:
            logger.info(f"\nAnalyzing table: {table}")
            logger.info("-" * 60)

            # Get columns for this table
            table_df = self.schema_explorer.df[
                self.schema_explorer.df['TABLE_NAME'] == table
            ]

            # Find FK candidates (columns ending in 'Id', excluding primary key)
            fk_candidates = [
                row['COLUMN_NAME']
                for _, row in table_df.iterrows()
                if row['COLUMN_NAME'].endswith('Id') and row['COLUMN_NAME'] != 'Id'
            ]

            if not fk_candidates:
                logger.info(f"  No FK candidates found")
                continue

            logger.info(f"  Found {len(fk_candidates)} FK candidates: {fk_candidates}")

            table_relationships = {}

            for fk_column in fk_candidates:
                logger.info(f"\n  Processing {fk_column}...")

                # Get data type
                col_row = table_df[table_df['COLUMN_NAME'] == fk_column].iloc[0]
                data_type = col_row['DATA_TYPE']

                relationship_info = self._discover_relationship(
                    table, fk_column, data_type
                )

                if relationship_info:
                    table_relationships[fk_column] = relationship_info
                    relationship_count += 1
                    logger.info(
                        f"    {fk_column} -> {relationship_info['target_table']} "
                        f"({relationship_info['discovery_method']}, "
                        f"confidence: {relationship_info['confidence']})"
                    )

            if table_relationships:
                metadata['relationships'][table] = table_relationships

            # Analyze data quality for this table
            quality_info = self._analyze_table_quality(table)
            if quality_info:
                metadata['data_quality'][table] = quality_info

        metadata['metadata']['csv_files_analyzed'] = len(tables_to_profile)
        metadata['metadata']['total_relationships'] = relationship_count

        logger.info("\n" + "="*60)
        logger.info(f"METADATA BUILD COMPLETE")
        logger.info(f"Tables analyzed: {len(tables_to_profile)}")
        logger.info(f"Relationships discovered: {relationship_count}")
        logger.info("="*60 + "\n")

        return metadata

    def save_metadata(self, metadata: Dict, output_path: str = "config/relationship_metadata.json") -> None:
        """Persist metadata to disk."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {out_path}")

    def _discover_relationship(
        self,
        source_table: str,
        source_column: str,
        data_type: str
    ) -> Optional[Dict]:
        """
        Discover FK relationship using three-tier approach.

        Tier 1: Check exceptions first
        Tier 2: Value-based profiling
        Tier 3: Convention-based inference

        Returns:
            Relationship info dictionary or None
        """
        # TIER 1: Check exceptions
        exception_target = get_exception_target(source_column, source_table)
        if exception_target:
            logger.info(f"    Using exception mapping")
            return {
                'target_table': exception_target,
                'target_column': 'Id',
                'discovery_method': 'exception',
                'confidence': 'HIGH',
                'match_rate': 1.0,  # Assumed perfect match for exceptions
                'data_type': data_type,
                'verified': False,  # Could be verified later
                'source': 'relationship_exceptions.py'
            }

        # TIER 2: Value-based profiling
        # Try convention first to get target table, then validate with data
        convention_target = source_column[:-2].lower()  # Remove 'Id' suffix

        if convention_target in self.schema_explorer.tables:
            # Table exists - validate with actual data
            logger.info(f"    Convention suggests: {convention_target}, validating with data...")

            profile_result = self.profiler.profile_relationship(
                source_table, source_column, convention_target
            )

            if profile_result['confidence'] in ['HIGH', 'MEDIUM']:
                logger.info(
                    f"    Value-based validation: {profile_result['confidence']} "
                    f"(match_rate: {profile_result['match_rate']:.2%})"
                )

                return {
                    'target_table': convention_target,
                    'target_column': 'Id',
                    'discovery_method': 'value_based',
                    'confidence': profile_result['confidence'],
                    'match_rate': profile_result['match_rate'],
                    'data_type': data_type,
                    'sample_size': profile_result['sample_size'],
                    'null_rate': profile_result.get('null_rate', 0.0),
                    'is_sampled': profile_result['is_sampled'],
                    'orphan_count': profile_result['orphan_count'],
                    'verified': True,
                    'warnings': profile_result['warnings']
                }

        # TIER 3: Convention-based (unverified)
        # If convention suggests a table that exists but we couldn't validate with data
        if convention_target in self.schema_explorer.tables:
            logger.info(f"    Using convention (unverified)")
            return {
                'target_table': convention_target,
                'target_column': 'Id',
                'discovery_method': 'convention',
                'confidence': 'LOW',
                'match_rate': None,
                'data_type': data_type,
                'verified': False,
                'warnings': ['Convention-based inference, not validated with data']
            }

        # Could not discover relationship
        logger.warning(f"    Could not discover relationship for {source_column}")
        return None

    def _analyze_table_quality(self, table_name: str) -> Optional[Dict]:
        """Analyze data quality for a table."""
        try:
            completeness = self.quality_analyzer.analyze_table_completeness(table_name)

            if 'error' in completeness:
                return None

            # Get null rates for columns with high nulls
            null_info = {}
            if completeness['columns_with_high_nulls']:
                null_rates = self.quality_analyzer.analyze_null_rates(table_name)
                for col in completeness['columns_with_high_nulls']:
                    null_info[col] = {
                        'null_rate': null_rates.get(col, 0.0),
                        'impact': 'May affect join coverage'
                    }

            return {
                'row_count': completeness['row_count'],
                'completeness_score': completeness['completeness_score'],
                'is_sample': completeness['is_sample'],
                'high_null_columns': null_info
            }

        except Exception as e:
            logger.error(f"Error analyzing quality for {table_name}: {e}")
            return None

    def validate_exceptions(self) -> Dict[str, Dict]:
        """
        Validate hardcoded exceptions against actual data.

        Returns:
            Validation report for each exception
        """
        logger.info("Validating hardcoded exceptions...")

        validation_report = {}

        for column_name, exception_info in RELATIONSHIP_EXCEPTIONS.items():
            logger.info(f"\nValidating: {column_name}")

            for source_table, target_table in exception_info['tables'].items():
                if source_table == '*':
                    continue  # Skip wildcard entries

                # Try to validate with data
                result = self.profiler.profile_relationship(
                    source_table, column_name, target_table
                )

                status = 'VALID' if result['confidence'] in ['HIGH', 'MEDIUM'] else 'INVALID'

                validation_report[f"{source_table}.{column_name}"] = {
                    'status': status,
                    'expected_target': target_table,
                    'match_rate': result['match_rate'],
                    'confidence': result['confidence'],
                    'warnings': result.get('warnings', [])
                }

                logger.info(
                    f"  {source_table}.{column_name} Â {target_table}: {status} "
                    f"(match_rate: {result['match_rate']:.2%})"
                )

        return validation_report

    def incremental_update(self, metadata_path: str) -> Dict:
        """
        Update only changed CSVs based on file timestamps.

        Args:
            metadata_path: Path to existing metadata.json file

        Returns:
            Updated metadata dictionary
        """
        logger.info("Performing incremental update...")

        # Load existing metadata
        if not Path(metadata_path).exists():
            logger.warning("No existing metadata found, performing full build")
            return self.build_metadata(force_refresh=True)

        with open(metadata_path) as f:
            existing_metadata = json.load(f)

        # Check which CSV files have been modified
        metadata_timestamp = datetime.fromisoformat(
            existing_metadata['metadata']['generated_at']
        )

        modified_tables = []
        for csv_file in self.data_dir.glob('*.csv'):
            file_mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
            if file_mtime > metadata_timestamp:
                table_name = csv_file.stem.replace('_sample', '')
                modified_tables.append(table_name)

        if not modified_tables:
            logger.info("No CSV files modified since last build")
            return existing_metadata

        logger.info(f"Re-profiling {len(modified_tables)} modified tables: {modified_tables}")

        # Re-profile only modified tables
        new_metadata = self.build_metadata(tables_to_profile=modified_tables)

        # Merge with existing metadata
        for table in modified_tables:
            if table in new_metadata['relationships']:
                existing_metadata['relationships'][table] = new_metadata['relationships'][table]
            if table in new_metadata['data_quality']:
                existing_metadata['data_quality'][table] = new_metadata['data_quality'][table]

        # Update metadata timestamp
        existing_metadata['metadata']['generated_at'] = datetime.now().isoformat()

        logger.info("Incremental update complete")
        return existing_metadata

    def save_metadata(self, metadata: Dict, output_path: str):
        """
        Save metadata to JSON file.

        Args:
            metadata: Metadata dictionary
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to: {output_path}")

    def export_report(self, metadata: Dict, format: str = 'markdown') -> str:
        """
        Generate human-readable relationship report.

        Args:
            metadata: Metadata dictionary
            format: Output format ('markdown' or 'text')

        Returns:
            Formatted report string
        """
        if format == 'markdown':
            return self._export_markdown_report(metadata)
        else:
            return self._export_text_report(metadata)

    def _export_markdown_report(self, metadata: Dict) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Relationship Discovery Report\n")
        lines.append(f"Generated: {metadata['metadata']['generated_at']}\n")
        lines.append(f"Tables analyzed: {metadata['metadata']['csv_files_analyzed']}")
        lines.append(f"Relationships discovered: {metadata['metadata']['total_relationships']}\n")

        lines.append("## Discovered Relationships\n")

        for table, relationships in sorted(metadata['relationships'].items()):
            lines.append(f"### {table}\n")
            lines.append("| Column | Target Table | Discovery Method | Confidence | Match Rate |")
            lines.append("|--------|--------------|------------------|------------|------------|")

            for column, info in sorted(relationships.items()):
                match_rate = f"{info['match_rate']:.1%}" if info.get('match_rate') else 'N/A'
                lines.append(
                    f"| {column} | {info['target_table']} | "
                    f"{info['discovery_method']} | {info['confidence']} | {match_rate} |"
                )

            lines.append("")

        return '\n'.join(lines)

    def _export_text_report(self, metadata: Dict) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("="*70)
        lines.append("RELATIONSHIP DISCOVERY REPORT")
        lines.append("="*70)
        lines.append(f"Generated: {metadata['metadata']['generated_at']}")
        lines.append(f"Tables analyzed: {metadata['metadata']['csv_files_analyzed']}")
        lines.append(f"Relationships discovered: {metadata['metadata']['total_relationships']}")
        lines.append("="*70 + "\n")

        for table, relationships in sorted(metadata['relationships'].items()):
            lines.append(f"\n{table}:")
            lines.append("-"*70)

            for column, info in sorted(relationships.items()):
                match_rate = f"{info['match_rate']:.1%}" if info.get('match_rate') else 'N/A'
                lines.append(
                    f"  {column} -> {info['target_table']} "
                    f"({info['discovery_method']}, {info['confidence']}, match: {match_rate})"
                )

        return '\n'.join(lines)


def main():
    """CLI entry for building metadata (supports flags or interactive prompts)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build relationship_metadata.json from CSVs and schema.",
        epilog="Tip: run without positional args for an interactive prompt."
    )
    parser.add_argument("data_dir", nargs="?", help="Path to directory containing CSV files")
    parser.add_argument("excel_schema", nargs="?", help="Path to schema Excel file")
    parser.add_argument("--force", action="store_true", help="Force re-profile even if metadata exists")
    parser.add_argument("--tables", nargs="+", help="Optional list of tables to profile")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for profiling")
    parser.add_argument("--enrich", action="store_true", help="Enable template/CSV enrichment in SchemaExplorer")
    parser.add_argument("--out", default="config/relationship_metadata.json", help="Output path for metadata JSON")
    args = parser.parse_args()

    interactive_mode = not args.data_dir or not args.excel_schema

    # Interactive prompt if required args missing
    if interactive_mode:
        print("Interactive mode (press Enter to accept defaults):")
        data_dir = input("Data directory (CSV files) [data]: ").strip() or "data"
        excel_schema = input("Schema Excel path [./AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx]: ").strip() \
            or "./AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx"
        sample_rows_input = input("Sample rows (int, blank for all) []: ").strip()
        sample_rows = int(sample_rows_input) if sample_rows_input else None
        enrich_input = input("Enable enrichment (templates/CSVs)? [y/N]: ").strip().lower()
        enrich = enrich_input == 'y'
        force_input = input("Force refresh profiling? [y/N]: ").strip().lower()
        force = force_input == 'y'
        tables_input = input("Limit to specific tables (comma-separated, blank for all) []: ").strip()
        tables = [t.strip() for t in tables_input.split(",") if t.strip()] if tables_input else None
        out_path = input("Output path [config/relationship_metadata.json]: ").strip() or "config/relationship_metadata.json"
    else:
        data_dir = args.data_dir
        excel_schema = args.excel_schema
        sample_rows = args.sample_rows
        enrich = args.enrich
        force = args.force
        tables = args.tables
        out_path = args.out

    print("\n=== METADATA BUILD CONFIG ===")
    print(f"Data dir:        {data_dir}")
    print(f"Schema file:     {excel_schema}")
    print(f"Sample rows:     {sample_rows if sample_rows is not None else 'all'}")
    print(f"Enrichment:      {'ON' if enrich else 'OFF'}")
    print(f"Force refresh:   {'YES' if force else 'NO'}")
    print(f"Tables filter:   {', '.join(tables) if tables else 'all'}")
    print(f"Output path:     {out_path}")
    print("=============================\n")

    builder = MetadataBuilder(
        data_dir,
        excel_schema,
        sample_rows=sample_rows,
        enable_enrichment=enrich,
        interactive=interactive_mode
    )
    metadata = builder.build_metadata(force_refresh=force, tables_to_profile=tables)
    builder.save_metadata(metadata, out_path)


if __name__ == '__main__':
    main()
