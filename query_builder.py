#!/usr/bin/env python3
"""
SPE-CDNI Query Builder
Interactive CLI for building SQL queries without SQL knowledge
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

from core.intent_interpreter import IntentInterpreter, ParsedIntent
from core.schema_navigator import SchemaNavigator
from generators.query_generator import QueryGenerator
from utils.data_catalog import DataCatalog
import re
from dataclasses import dataclass

# Transaction status handling for gos/transaction tables
TRANSACTION_STATUS_CONFIG = {
    'gostransaction': {
        'column': 'StatusCode',
        'defaults': ['B', 'I'],  # Booked/Processed
    },
    'transaction': {
        'column': 'Status',
        'defaults': None,  # Unknown integer meanings; do not force
    }
}

# Columns we should not auto-aggregate due to bad export formatting in sample CSVs
UNSAFE_AGG_COLUMNS = {
    'gostransaction': {'Amount', 'NewBalance'},
}


@dataclass
class TemplateHint:
    base_table: Optional[str] = None
    join_tables: List[str] = None
    group_by: List[str] = None
    where_columns: List[str] = None


class TemplateLibrary:
    """Lightweight loader for known-good SQL templates."""

    def __init__(self, base_dir: str = "saved_queries"):
        self.base = Path(base_dir)
        self.base.mkdir(exist_ok=True)
        self.hints_cache: Dict[Path, TemplateHint] = {}

    def list_templates(self):
        return sorted(self.base.glob("*.sql"))

    def read_template(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _parse_template(self, path: Path) -> TemplateHint:
        if path in self.hints_cache:
            return self.hints_cache[path]

        text = self.read_template(path)
        base_table = None
        join_tables: List[str] = []
        group_by: List[str] = []
        where_columns: List[str] = []

        # FROM table
        m = re.search(r"FROM\s+`?([a-zA-Z0-9_]+)`?", text, flags=re.IGNORECASE)
        if m:
            base_table = m.group(1).lower()

        # JOIN tables
        for jm in re.finditer(r"JOIN\s+`?([a-zA-Z0-9_]+)`?", text, flags=re.IGNORECASE):
            join_tables.append(jm.group(1).lower())

        # GROUP BY columns
        gm = re.search(r"GROUP\s+BY\s+([^\n;]+)", text, flags=re.IGNORECASE)
        if gm:
            cols = gm.group(1)
            group_by = [c.strip() for c in cols.split(",") if c.strip()]

        # WHERE columns (naive)
        for wm in re.finditer(r"WHERE\s+([^\n;]+)", text, flags=re.IGNORECASE):
            conds = wm.group(1).split("AND")
            for cond in conds:
                col_match = re.match(r"`?([a-zA-Z0-9_]+)`?\.", cond.strip())
                if col_match:
                    where_columns.append(col_match.group(1).lower())

        hint = TemplateHint(
            base_table=base_table,
            join_tables=join_tables,
            group_by=group_by,
            where_columns=where_columns,
        )
        self.hints_cache[path] = hint
        return hint

    def match_intent(self, intent: ParsedIntent) -> Optional[Path]:
        """Return a template path if intent matches a known pattern."""
        entities = set(intent.entities)

        # Waste by country per year (aggregation, year, country filter)
        if (
            'waste' in entities
            and intent.filters.get('country')
            and (intent.time_dimension == 'year' or intent.intent_type in ('aggregation', 'trend'))
        ):
            candidate = self.base / "waste_total_by_country_year.sql"
            if candidate.exists():
                return candidate

        # Processed GOS transactions by facility per year
        if (
            ('transaction' in entities or 'fuel' in entities)
            and ('facility' in entities)
        ):
            candidate = self.base / "gos_processed_by_facility.sql"
            if candidate.exists():
                return candidate

        return None

    def match_hints(self, intent: ParsedIntent) -> Optional[TemplateHint]:
        path = self.match_intent(intent)
        if not path:
            return None
        return self._parse_template(path)


class QueryBuilderCLI:
    """Main CLI interface for the query builder"""

    def __init__(self, schema_path: str):
        """
        Initialize the query builder.

        Args:
            schema_path: Path to the schema Excel file
        """
        print("Initializing SPE-CDNI Query Builder...")
        self.catalog = DataCatalog()
        self.interpreter = IntentInterpreter(data_catalog=self.catalog)
        self.navigator = SchemaNavigator(schema_path)
        self.generator = QueryGenerator()
        self.templates = TemplateLibrary()
        self.active_template_hint: Optional[TemplateHint] = None
        print("[OK] Ready!\n")

    def start(self):
        """Start the interactive CLI"""
        self._print_welcome()

        while True:
            try:
                # Get user input
                question = input("\nWhat would you like to analyze?\n> ").strip()

                if not question:
                    continue

                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if question.lower() in ['help', 'h']:
                    self._print_help()
                    continue

                if question.lower() in ['templates', 'template', 't']:
                    self._print_templates()
                    continue

                # Process the question
                self._process_question(question)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                print("Please try again or type 'help' for assistance.")

    def _print_welcome(self):
        """Print welcome message"""
        print("=" * 80)
        print("SPE-CDNI QUERY BUILDER")
        print("=" * 80)
        print("\nBuild SQL queries by describing what you want to analyze in natural language.")
        print("\nExamples:")
        print("  - Total waste disposal for Belgium per year")
        print("  - Geographic distribution of fuel transactions by Belgian vessels")
        print("  - Count of transactions by vessel type")
        print("\nCommands:")
        print("  help  - Show detailed help")
        print("  quit  - Exit the query builder")
        print()

    def _print_help(self):
        """Print detailed help"""
        print("\n" + "=" * 80)
        print("QUERY BUILDER HELP")
        print("=" * 80)
        print("\nHow to use:")
        print("1. Describe your analysis question in natural language")
        print("2. Answer clarifying questions if asked")
        print("3. Review and refine the generated SQL")
        print("4. Copy the SQL to the web UI: https://new.spe-cdni.org/Reports")
        print("\nSupported query types:")
        print("  - Aggregations: totals, counts, averages")
        print("  - Filtering: by country, date range, status, etc.")
        print("  - Grouping: by year, month, country, type, etc.")
        print("  - Listings: show all records matching criteria")
        print("\nKey entities you can ask about:")
        print("  - Vessels/Ships (schepen)")
        print("  - Transactions (transacties)")
        print("  - Waste collection (afval/verwijdering)")
        print("  - Accounts/ECO-IDs")
        print("  - Countries/Geographic data")
        print("  - Fuel types/Facilities")
        print()

    def _process_question(self, question: str):
        """
        Main query processing workflow.

        Args:
            question: User's natural language question
        """
        print("\n" + "=" * 80)
        print("BUILDING QUERY")
        print("=" * 80)

        # Step 1: Parse intent
        print("\n[1] Understanding your question...")
        self.active_template_hint = None
        intent = self.interpreter.parse_question(question)

        print(f"\nI detected:")
        print(f"  Entities: {', '.join(intent.entities) if intent.entities else 'None identified'}")
        print(f"  Intent: {intent.intent_type}")
        if intent.aggregation:
            print(f"  Aggregation: {intent.aggregation}")
        if intent.time_dimension:
            print(f"  Time dimension: per {intent.time_dimension}")
        if intent.filters:
            print(f"  Filters: {intent.filters}")
            self._print_filter_tips(intent)

        # Offer template if available; also capture hints for generation
        if self._maybe_use_template(intent, question):
            # Template used; skip dynamic build
            return
        else:
            self.active_template_hint = self.templates.match_hints(intent)

        # Check confidence
        if intent.confidence < 0.5:
            print(f"\n[WARNING] Low confidence ({intent.confidence:.0%}). Results may need refinement.")

        # Handle clarifications
        if intent.clarifications:
            print("\n" + "-" * 80)
            print("I need some clarification:")
            for clarification in intent.clarifications:
                print(f"\n{clarification}")

            response = input("\nPlease provide clarification (or 'skip' to continue anyway): ").strip()
            if response.lower() == 'skip':
                print("Continuing with best guess...")
            else:
                # For MVP, we'll just acknowledge and continue
                # In future, parse the clarification response
                print("Thank you for the clarification.")

        # Step 2: Identify tables
        print("\n[2] Identifying relevant tables...")
        entity_tables = self.navigator.find_tables_for_entities(intent.entities)

        if not entity_tables:
            print("[ERROR] Could not identify relevant tables from your question.")
            print("Try being more specific about what data you need (vessels, transactions, waste, etc.)")
            return

        print(f"\nRelevant tables found:")
        for entity, tables in entity_tables.items():
            print(f"  {entity}: {', '.join(tables)}")

        # Choose base table, prefer template hint if available
        primary_entity = intent.entities[0] if intent.entities else None
        base_table = None
        if self.active_template_hint and self.active_template_hint.base_table:
            hinted = self.active_template_hint.base_table
            if hinted in self.navigator.tables:
                base_table = hinted
                print(f"\nUsing template base table '{base_table}'.")
        if not base_table:
            if not primary_entity:
                print("\n[ERROR] Could not determine primary data source.")
                return
            primary_tables = entity_tables[primary_entity]
            base_table = primary_tables[0]

        print(f"\nUsing '{base_table}' as the main table.")

        # Step 3: Handle joins if multiple entities
        if len(intent.entities) > 1:
            print("\n[3] Finding join paths...")
            if not self._apply_template_joins(base_table):
                # For MVP, we'll do a simple path finding between first two entities
                target_entity = intent.entities[1]
                target_tables = entity_tables[target_entity]
                target_table = target_tables[0]

                paths = self.navigator.find_join_path(base_table, target_table, max_depth=3)

                if paths:
                    path = paths[0]  # Use shortest path
                    print(f"Join path: {' -> '.join(path)}")

                    # Build joins
                    join_info = self.navigator.build_join_chain(path)
                    self.generator.build_join_chain_from_path(path, join_info)
                else:
                    print(f"[WARNING] No direct join path found between {base_table} and {target_table}")
                    print("Continuing with base table only...")
                    self.generator.set_base_table(base_table)
        else:
            # Single entity - just set base table
            self.generator.set_base_table(base_table)

        # Step 4: Build SELECT clause
        print("\n[4] Building SELECT clause...")
        self._build_select_clause(intent, base_table)

        # Step 5: Build WHERE clause
        if intent.filters or intent.intent_type == 'filtering':
            print("\n[5] Adding filters...")
            self._build_where_clause(intent, base_table)

        # Step 6: Build GROUP BY if needed
        if intent.intent_type in ['aggregation', 'trend'] or intent.time_dimension:
            print("\n[6] Adding grouping...")
            self._build_group_by(intent, base_table)
            self._apply_template_group_by()

        # Step 7: Generate SQL
        print("\n" + "=" * 80)
        print("GENERATED QUERY")
        print("=" * 80)

        try:
            sql = self.generator.generate_with_comments(description=question)
            print(f"\n{sql}\n")

            # Show explanation
            print("\n" + "-" * 80)
            print(self.generator.explain_query())
            print("-" * 80)

            # Show complexity
            complexity = self.generator.get_estimated_complexity()
            print(f"\nQuery complexity: {complexity.upper()}")

            print("\n[NEXT STEPS]")
            print("1. Review the query above")
            print("2. Copy the SQL (between the comments)")
            print("3. Go to: https://new.spe-cdni.org/Reports")
            print("4. Paste and execute the query")
            print("5. Download results as CSV")

        except ValueError as e:
            print(f"\n[ERROR] Could not generate query: {e}")
            print("The query requirements may need refinement.")

        finally:
            # Reset for next query
            self.generator.reset()

    def _build_select_clause(self, intent: ParsedIntent, base_table: str):
        """Build the SELECT clause based on intent"""
        alias = self.generator.state.table_aliases[base_table]

        if intent.aggregation:
            # Aggregation query with safe column selection
            agg = intent.aggregation
            if agg == 'COUNT':
                self.generator.add_select_column(f"COUNT(DISTINCT {alias}.Id)", "count")
                return

            # Prefer schema-derived numeric suggestions, skipping unsafe columns
            table_suggestions = self.navigator.suggest_aggregation_columns(base_table) or []
            chosen_col = None
            for suggestion in table_suggestions:
                col = suggestion['column']
                if col in UNSAFE_AGG_COLUMNS.get(base_table, set()):
                    # Warn once per unsafe column
                    print(f"  Skipping unsafe aggregation column '{col}' for {base_table} (bad CSV formatting).")
                    continue
                if agg in suggestion.get('suggested_agg', []):
                    chosen_col = col
                    break

            if chosen_col:
                print(f"  Using {agg} on {base_table}.{chosen_col} (schema-derived numeric column).")
                self.generator.add_select_column(f"{agg}({alias}.{chosen_col})", agg.lower())
            else:
                print(f"  No safe numeric column found for {agg}; defaulting to COUNT.")
                self.generator.add_select_column(f"COUNT(DISTINCT {alias}.Id)", "count")
        else:
            # Simple listing - select a few key columns
            table_info = self.navigator.get_table_info(base_table)
            if table_info:
                # Select first 5 columns for MVP
                for col in table_info['columns'][:5]:
                    self.generator.add_select_column(f"{alias}.{col['name']}")

    def _build_where_clause(self, intent: ParsedIntent, base_table: str):
        """Build WHERE clause from filters"""
        alias = self.generator.state.table_aliases[base_table]

        # Add country filter if present
        if 'country' in intent.filters:
            country = intent.filters['country']
            # Try to find country reference in joins
            if 'country' in [t for t, _ in self.generator.state.joined_tables]:
                country_alias = self.generator.state.table_aliases['country']
                self.generator.add_where_condition(f"{country_alias}.Name = '{country}'")
            print(f"  Filter: country = {country}")

        # Add status filter (default or user requested)
        status_tables = [t for t in self.generator.state.table_aliases if t in TRANSACTION_STATUS_CONFIG]
        status_filter = intent.filters.get('status_codes')
        defaults_applied = False
        target_table = status_tables[0] if status_tables else None

        if not status_filter and target_table:
            default_codes = TRANSACTION_STATUS_CONFIG[target_table]['defaults']
            if default_codes:
                status_filter = default_codes
                defaults_applied = True

        if status_filter and target_table:
            col_name = TRANSACTION_STATUS_CONFIG[target_table]['column']
            alias = self.generator.state.table_aliases[target_table]
            if len(status_filter) == 1:
                self.generator.add_where_condition(f"{alias}.{col_name} = '{status_filter[0]}'")
            else:
                codes = "', '".join(status_filter)
                self.generator.add_where_condition(f"{alias}.{col_name} IN ('{codes}')")
            label = "default" if defaults_applied else "requested"
            print(f"  Filter: status codes ({label}) -> {status_filter}")

        # Add waste type filter if we can reach wastetype table
        if 'waste_type' in intent.filters:
            waste_name = intent.filters['waste_type']
            joined_tables = set(self.generator.state.table_aliases.keys())
            if 'wastetype' not in joined_tables:
                joined = self._ensure_join_for_table(base_table, 'wastetype')
            else:
                joined = True
            if joined and 'wastetype' in self.generator.state.table_aliases:
                wt_alias = self.generator.state.table_aliases['wastetype']
                self.generator.add_where_condition(f"{wt_alias}.Name = '{waste_name}'")
                print(f"  Filter: waste type = {waste_name}")
            else:
                print(f"  [WARNING] Could not add waste type filter (missing join to wastetype).")

        # Add year filter if present
        if 'year' in intent.filters:
            year = intent.filters['year']
            # Find date column in base table
            table_info = self.navigator.get_table_info(base_table)
            date_cols = [col for col in table_info['columns']
                        if any(word in col['name'].lower() for word in ['date', 'time'])]
            if date_cols:
                date_col = date_cols[0]['name']
                self.generator.add_where_condition(f"YEAR({alias}.{date_col}) = {year}")
                print(f"  Filter: year = {year}")

    def _build_group_by(self, intent: ParsedIntent, base_table: str):
        """Build GROUP BY clause"""
        alias = self.generator.state.table_aliases[base_table]

        # Add time dimension grouping
        if intent.time_dimension:
            table_info = self.navigator.get_table_info(base_table)
            date_cols = [col for col in table_info['columns']
                        if any(word in col['name'].lower() for word in ['date', 'time'])]

            if date_cols:
                date_col = date_cols[0]['name']
                if intent.time_dimension == 'year':
                    self.generator.add_select_column(f"YEAR({alias}.{date_col})", "year")
                    self.generator.add_group_by(f"YEAR({alias}.{date_col})")
                    print(f"  Group by: year")
                elif intent.time_dimension == 'month':
                    self.generator.add_select_column(f"YEAR({alias}.{date_col})", "year")
                    self.generator.add_select_column(f"MONTH({alias}.{date_col})", "month")
                    self.generator.add_group_by(f"YEAR({alias}.{date_col}), MONTH({alias}.{date_col})")
                    print(f"  Group by: year, month")

        # Add country grouping if country is in joins
        if 'country' in [t for t, _ in self.generator.state.joined_tables]:
            country_alias = self.generator.state.table_aliases['country']
            self.generator.add_select_column(f"{country_alias}.Name", "country")
            self.generator.add_group_by(f"{country_alias}.Name")
            print(f"  Group by: country")

        # Add waste type grouping if joined
        if 'wastetype' in self.generator.state.table_aliases:
            wt_alias = self.generator.state.table_aliases['wastetype']
            self.generator.add_select_column(f"{wt_alias}.Name", "waste_type")
            self.generator.add_group_by(f"{wt_alias}.Name")
            print(f"  Group by: waste type")

    def _ensure_join_for_table(self, base_table: str, target_table: str) -> bool:
        """Attempt to add a join to target_table if a direct relationship exists."""
        if target_table in self.generator.state.table_aliases:
            return True

        condition = self.navigator.get_join_condition(base_table, target_table)
        if not condition:
            return False

        # Ensure alias exists for target table before replacement
        if target_table not in self.generator.state.table_aliases:
            self.generator.state.table_aliases[target_table] = self.generator._create_alias(target_table)

        condition_with_aliases = self.generator._apply_aliases_to_condition(condition)
        self.generator.add_join(target_table, condition_with_aliases)
        return True

    def _print_filter_tips(self, intent: ParsedIntent):
        """Surface lookup values from CSV catalog to guide user filters."""
        if intent.filters.get('country') and self.catalog.country_records():
            country = intent.filters['country']
            suggestions = self.catalog.closest_countries(country)
            if suggestions:
                print(f"    Country tips: closest matches -> {', '.join(suggestions)}")
        if 'waste' in intent.entities and not intent.filters.get('waste_type'):
            waste_labels = self.catalog.waste_type_with_units()[:5]
            if waste_labels:
                print(f"    Waste types you can use: {', '.join(waste_labels)}")
        if 'facility' in intent.entities:
            facilities = self.catalog.facility_names(limit=5)
            if facilities:
                print(f"    Example facilities: {', '.join(facilities)}")

    def _maybe_use_template(self, intent: ParsedIntent, question: str) -> bool:
        """Check if a known template fits the intent and optionally use it."""
        path = self.templates.match_intent(intent)
        if not path:
            return False

        print("\n[Template Suggestion]")
        print(f"Found known-good query: {path.name}")
        content = self.templates.read_template(path)
        print("\n--- TEMPLATE START ---")
        print(content.strip())
        print("--- TEMPLATE END ---\n")

        resp = input("Use this template? [y/N]: ").strip().lower()
        if resp == 'y':
            print("\nUsing template SQL. Edit inline placeholders (e.g., country/facility) if needed before running in SPE-CDNI.\n")
            return True
        return False

    def _apply_template_joins(self, base_table: str) -> bool:
        """If template hints specify join order, apply them to the generator."""
        if not self.active_template_hint or not self.active_template_hint.join_tables:
            return False
        path = [base_table] + [t for t in self.active_template_hint.join_tables if t in self.navigator.tables]
        if len(path) <= 1:
            return False
        join_info = self.navigator.build_join_chain(path)
        if not join_info:
            return False
        print(f"Using template join path: {' -> '.join(path)}")
        self.generator.build_join_chain_from_path(path, join_info)
        return True

    def _apply_template_group_by(self):
        """Apply template group-by expressions to SELECT/GROUP BY."""
        if not self.active_template_hint or not self.active_template_hint.group_by:
            return
        for expr in self.active_template_hint.group_by:
            expr = expr.strip()
            if not expr:
                continue
            aliased_expr = self.generator._apply_aliases_to_condition(expr)
            if aliased_expr not in self.generator.state.group_by_columns:
                self.generator.add_group_by(aliased_expr)
                if aliased_expr not in self.generator.state.select_columns:
                    self.generator.add_select_column(aliased_expr)
                print(f"  Group by (template): {aliased_expr}")

    def _print_templates(self):
        """List available templates."""
        templates = self.templates.list_templates()
        if not templates:
            print("\nNo templates found in saved_queries/. Add .sql files to enable template reuse.\n")
            return
        print("\nAvailable templates:")
        for p in templates:
            print(f"  - {p.name}")
        print()


def main():
    """Main entry point"""
    # Path to schema file
    schema_path = './AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx'

    # Check if schema file exists
    if not Path(schema_path).exists():
        print(f"[ERROR] Schema file not found at {schema_path}")
        print("Please ensure the Excel schema file is in the current directory.")
        sys.exit(1)

    try:
        # Initialize and start CLI
        cli = QueryBuilderCLI(schema_path)
        cli.start()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
