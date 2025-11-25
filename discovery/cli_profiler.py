"""
Command-line tool for relationship profiling.

Usage:
    python -m discovery.cli_profiler --profile-all
    python -m discovery.cli_profiler --profile-table gostransaction
    python -m discovery.cli_profiler --validate-exceptions
    python -m discovery.cli_profiler --export-report relationship_report.md
    python -m discovery.cli_profiler --incremental
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.metadata_builder import MetadataBuilder
from discovery.relationship_profiler import RelationshipProfiler
from discovery.data_quality_analyzer import DataQualityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CDNI Relationship Profiler - Discover FK relationships from CSV data',
        epilog='For more information, see the implementation plan.'
    )

    # Commands
    parser.add_argument(
        '--profile-all',
        action='store_true',
        help='Profile all CSV files and build complete metadata'
    )

    parser.add_argument(
        '--profile-table',
        type=str,
        metavar='TABLE',
        help='Profile specific table only (faster than --profile-all)'
    )

    parser.add_argument(
        '--validate-exceptions',
        action='store_true',
        help='Validate hardcoded exceptions against actual CSV data'
    )

    parser.add_argument(
        '--export-report',
        type=str,
        metavar='PATH',
        help='Export relationship report to file (markdown format)'
    )

    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Incremental update - only re-profile changed CSV files'
    )

    parser.add_argument(
        '--quality-report',
        type=str,
        metavar='TABLE',
        help='Generate data quality report for specific table'
    )

    # Configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing CSV files (default: data)'
    )

    parser.add_argument(
        '--schema-file',
        type=str,
        default='AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx',
        help='Path to schema Excel file (default: AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='config/relationship_metadata.json',
        help='Output path for metadata JSON (default: config/relationship_metadata.json)'
    )

    args = parser.parse_args()

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    schema_file = Path(args.schema_file)
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        sys.exit(1)

    # Initialize builder
    try:
        builder = MetadataBuilder(
            data_dir=str(data_dir),
            excel_schema=str(schema_file)
        )
    except Exception as e:
        logger.error(f"Failed to initialize metadata builder: {e}")
        sys.exit(1)

    # Execute commands
    try:
        if args.profile_all:
            profile_all(builder, args.output)

        elif args.profile_table:
            profile_table(builder, args.profile_table, args.output)

        elif args.validate_exceptions:
            validate_exceptions(builder)

        elif args.incremental:
            incremental_update(builder, args.output)

        elif args.quality_report:
            quality_report(args.data_dir, args.quality_report)

        elif args.export_report:
            export_report(args.output, args.export_report)

        else:
            parser.print_help()
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def profile_all(builder: MetadataBuilder, output_path: str):
    """Profile all tables and save metadata."""
    logger.info("Starting full profiling of all tables...")

    metadata = builder.build_metadata(force_refresh=True)

    # Save metadata
    builder.save_metadata(metadata, output_path)

    # Print summary
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print(f"Tables analyzed: {metadata['metadata']['csv_files_analyzed']}")
    print(f"Relationships discovered: {metadata['metadata']['total_relationships']}")
    print(f"Metadata saved to: {output_path}")
    print("="*70)


def profile_table(builder: MetadataBuilder, table_name: str, output_path: str):
    """Profile a single table."""
    logger.info(f"Profiling table: {table_name}")

    metadata = builder.build_metadata(tables_to_profile=[table_name])

    if table_name in metadata['relationships']:
        print(f"\nRelationships found in {table_name}:")
        for column, info in metadata['relationships'][table_name].items():
            match_rate = f"{info['match_rate']:.1%}" if info.get('match_rate') else 'N/A'
            print(
                f"  {column} -> {info['target_table']} "
                f"({info['discovery_method']}, {info['confidence']}, match: {match_rate})"
            )

            if info.get('warnings'):
                for warning in info['warnings']:
                    print(f"    WARNING: {warning}")
    else:
        print(f"No FK relationships found in {table_name}")


def validate_exceptions(builder: MetadataBuilder):
    """Validate hardcoded exceptions."""
    logger.info("Validating hardcoded exceptions...")

    report = builder.validate_exceptions()

    print("\n" + "="*70)
    print("EXCEPTION VALIDATION REPORT")
    print("="*70)

    valid_count = sum(1 for r in report.values() if r['status'] == 'VALID')
    total_count = len(report)

    for key, result in sorted(report.items()):
        status_symbol = "" if result['status'] == 'VALID' else ""
        print(
            f"{status_symbol} {key} Ã¯Â¿Â½ {result['expected_target']}: "
            f"{result['status']} (match_rate: {result['match_rate']:.2%})"
        )

        if result.get('warnings'):
            for warning in result['warnings']:
                print(f"    Ã¯Â¿Â½ {warning}")

    print("="*70)
    print(f"Valid: {valid_count}/{total_count}")
    print("="*70)


def incremental_update(builder: MetadataBuilder, metadata_path: str):
    """Perform incremental update."""
    logger.info("Performing incremental update...")

    updated_metadata = builder.incremental_update(metadata_path)

    # Save updated metadata
    builder.save_metadata(updated_metadata, metadata_path)

    print("\n" + "="*70)
    print("INCREMENTAL UPDATE COMPLETE")
    print("="*70)
    print(f"Metadata updated: {metadata_path}")
    print("="*70)


def quality_report(data_dir: str, table_name: str):
    """Generate data quality report for a table."""
    analyzer = DataQualityAnalyzer(data_dir)
    report = analyzer.generate_quality_report(table_name)
    print(report)


def export_report(metadata_path: str, report_path: str):
    """Export relationship report."""
    import json

    if not Path(metadata_path).exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Determine format from file extension
    format_type = 'markdown' if report_path.endswith('.md') else 'text'

    # Create a temporary builder just for export
    from discovery.metadata_builder import MetadataBuilder
    builder = MetadataBuilder('data', 'AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx')

    report_content = builder.export_report(metadata, format_type)

    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Report exported to: {report_path}")


if __name__ == '__main__':
    main()
