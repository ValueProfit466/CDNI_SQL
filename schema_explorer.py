#!/usr/bin/env python3
"""
SPE-CDNI Schema Explorer
Interactive tool to understand table relationships and build query paths
"""

import pandas as pd
from typing import List, Set, Dict, Tuple
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
import re

class OutputTee:
    """
    A class that writes output to multiple destinations simultaneously.
    This is named after the Unix 'tee' command which reads from standard input
    and writes to both standard output and one or more files.
    """
    
    def __init__(self, output_dir: str = "schema_analysis_output"):
        """
        Initialize the output handler with a base directory for storing files.
        
        The output_dir parameter specifies where analysis files will be saved.
        We create this directory if it does not already exist, which means the
        first time you run the tool it will set up its own workspace.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of output
        # This organizational structure keeps related analyses together
        (self.output_dir / "analyze").mkdir(exist_ok=True)
        (self.output_dir / "explore").mkdir(exist_ok=True)
        (self.output_dir / "path").mkdir(exist_ok=True)
        (self.output_dir / "list").mkdir(exist_ok=True)
        
        # Keep track of the current file we are writing to
        # This will be None when we are not actively saving output
        self.current_file: Optional[TextIO] = None
        self.current_filepath: Optional[Path] = None
    
    def start_capture(self, command_type: str, *args):
        """
        Begin capturing output to a new file.
        
        The command_type tells us which subdirectory to use (analyze, explore, etc.)
        The args are the parameters to the command, which we use in the filename.
        
        For example, if you run "analyze vessel", this method will create a file
        like "schema_analysis_output/analyze/vessel_20241121_143022.txt"
        """
        # Generate a unique filename using timestamp and command parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a descriptive filename from the command arguments
        # We join the arguments with underscores and clean up any invalid characters
        if args:
            # Convert arguments to strings and join them
            arg_str = "_".join(str(arg) for arg in args)
            # Remove or replace characters that are invalid in filenames
            arg_str = arg_str.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = f"{arg_str}_{timestamp}.txt"
        else:
            filename = f"output_{timestamp}.txt"
        
        # Construct the full path using the appropriate subdirectory
        self.current_filepath = self.output_dir / command_type / filename
        
        # Open the file for writing
        # We use 'w' mode which creates a new file or truncates an existing one
        # The encoding='utf-8' ensures proper handling of special characters
        self.current_file = open(self.current_filepath, 'w', encoding='utf-8')
        
        # Write a header to the file so readers know what they are looking at
        header = f"Schema Explorer Output\n"
        header += f"Command: {command_type} {' '.join(str(arg) for arg in args)}\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 100 + "\n\n"
        self.current_file.write(header)
    
    def write(self, text: str):
        """
        Write text to both the console and the current file (if one is open).
        
        This is the core of the Tee functionality. Every piece of output goes
        through this method, which ensures it appears both on screen and in the file.
        """
        # Always write to the console so users see immediate feedback
        sys.stdout.write(text)
        sys.stdout.flush()  # Ensure the output appears immediately
        
        # If we are currently capturing to a file, write there too
        if self.current_file is not None:
            self.current_file.write(text)
            self.current_file.flush()  # Ensure the file is updated immediately
    
    def stop_capture(self):
        """
        Stop capturing output and close the current file.
        
        This is called when a command finishes executing. It ensures the file
        is properly closed and saved, and it tells the user where to find it.
        """
        if self.current_file is not None:
            # Write a footer to mark the end of the output
            self.current_file.write("\n" + "=" * 100 + "\n")
            self.current_file.write(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Close the file to ensure all data is written and resources are released
            self.current_file.close()
            
            # Let the user know where the output was saved
            # This feedback is important so users know the feature is working
            print(f"\n📄 Output saved to: {self.current_filepath}")
            
            # Reset our tracking variables for the next command
            self.current_file = None
            self.current_filepath = None
    
    def print(self, *args, **kwargs):
        """
        A replacement for Python's built-in print function.
        
        This method accepts the same arguments as print() but routes the output
        through our write() method so it goes to both console and file.
        """
        # Convert all arguments to strings and join them with spaces
        # This mimics how the built-in print function works
        output = ' '.join(str(arg) for arg in args)
        
        # Add a newline at the end unless the user specified end=''
        end = kwargs.get('end', '\n')
        
        # Write the complete output through our tee mechanism
        self.write(output + end)

class SchemaExplorer:
    """Interactive tool for exploring database table relationships"""
    
    def __init__(
        self,
        excel_path: str,
        enable_file_output: bool = True,
        template_dir: str = "saved_queries",
        data_dir: str = "data",
        enable_enrichment: bool = False
    ):
        """
        Load the schema from Excel file and optionally enable file output.
        
        The enable_file_output parameter allows users to turn off file saving
        if they want, which is useful for quick queries where you do not need
        to keep the output.
        """
        self.df = pd.read_excel(excel_path)
        self.tables = set(self.df['TABLE_NAME'].unique())
        self.relationships = self._build_relationship_map()
        self.template_dir = Path(template_dir)
        self.data_dir = Path(data_dir)
        self.enrichment_enabled = enable_enrichment
        if enable_enrichment:
            self._enrich_relationships_from_templates()
            self._enrich_relationships_from_csv()
        
        # Initialize the output handler
        # We make this optional so the tool still works even if file output fails
        self.output_enabled = enable_file_output
        if enable_file_output:
            try:
                self.output = OutputTee()
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize file output: {e}")
                print("   Continuing with console output only.")
                self.output_enabled = False
        
        # Keep a reference to the original print function
        # This is useful for system messages that should not be captured
        self._original_print = print
    
    def _print(self, *args, **kwargs):
        """
        Internal print method that routes through the output handler if enabled.
        
        This method is used throughout the class instead of calling print directly.
        It gives us a single point of control over where output goes.
        """
        if self.output_enabled and hasattr(self, 'output'):
            self.output.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    def _build_relationship_map(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Build a map of foreign key relationships
        Returns: {table_name: [(fk_column, target_table, data_type), ...]}
        """
        rel_map = {}
        
        for table in self.tables:
            table_df = self.df[self.df['TABLE_NAME'] == table]
            fk_columns = []
            
            for _, row in table_df.iterrows():
                col_name = row['COLUMN_NAME']
                data_type = row['DATA_TYPE']
                
                # Identify foreign keys by naming pattern
                if col_name.endswith('Id') and col_name != 'Id':
                    # Infer target table name
                    target_table = col_name[:-2].lower()  # Remove 'Id' suffix
                    
                    # Check if target table exists
                    if target_table in self.tables:
                        fk_columns.append((col_name, target_table, data_type))
            
            if fk_columns:
                rel_map[table] = fk_columns
        
        return rel_map
    
    # Now update each method to use self._print instead of print
    # I will show you the pattern with the explore_table method
    
    def explore_table(self, table_name: str) -> None:
        """Display comprehensive information about a table"""
        
        # Start capturing output if file output is enabled
        if self.output_enabled and hasattr(self, 'output'):
            self.output.start_capture('explore', table_name)
        
        try:
            if table_name not in self.tables:
                self._print(f"❌ Table '{table_name}' not found in schema")
                self._print(f"Did you mean one of these? {self._find_similar_tables(table_name)}")
                return
            
            self._print(f"\n{'='*100}")
            self._print(f"TABLE: {table_name}")
            self._print(f"{'='*100}")
            
            # Show all columns
            table_df = self.df[self.df['TABLE_NAME'] == table_name]
            self._print(f"\n📋 COLUMNS ({len(table_df)}):")
            self._print(f"{'Column Name':<50} {'Data Type':<20} {'Notes':<30}")
            self._print(f"{'-'*100}")
            
            for _, row in table_df.iterrows():
                col_name = row['COLUMN_NAME']
                data_type = row['DATA_TYPE']
                
                notes = ""
                if col_name == 'Id':
                    notes = "🔑 PRIMARY KEY"
                elif col_name.endswith('Id'):
                    notes = "🔗 Likely FOREIGN KEY"
                elif 'DateTime' in col_name:
                    notes = "📅 Date/Time field"
                
                self._print(f"{col_name:<50} {data_type:<20} {notes:<30}")
            
            # Show foreign key relationships
            if table_name in self.relationships:
                self._print(f"\n🔗 OUTGOING RELATIONSHIPS (Foreign Keys in this table):")
                self._print(f"{'Column':<30} {'References Table':<30} {'Exists?':<10}")
                self._print(f"{'-'*80}")
                
                for fk_col, target, dtype in self.relationships[table_name]:
                    exists = "✅" if target in self.tables else "❌"
                    self._print(f"{fk_col:<30} {target:<30} {exists:<10}")
            
            # Show incoming relationships
            incoming = self._find_incoming_relationships(table_name)
            if incoming:
                self._print(f"\n🔗 INCOMING RELATIONSHIPS (Tables that reference this table):")
                self._print(f"{'From Table':<30} {'Via Column':<30}")
                self._print(f"{'-'*80}")
                
                for from_table, via_column in incoming:
                    self._print(f"{from_table:<30} {via_column:<30}")
        
        finally:
            # Always stop capturing, even if an error occurred
            # The try-finally pattern ensures cleanup happens no matter what
            if self.output_enabled and hasattr(self, 'output'):
                self.output.stop_capture()
            
    
    def _find_similar_tables(self, partial_name: str) -> List[str]:
        """Find tables with similar names"""
        partial_lower = partial_name.lower()
        similar = [t for t in self.tables if partial_lower in t.lower()]
        return similar[:5]
    
    def _find_incoming_relationships(self, target_table: str) -> List[Tuple[str, str]]:
        """Find all tables that have foreign keys pointing to target_table"""
        incoming = []
        
        for source_table, fk_list in self.relationships.items():
            for fk_col, target, _ in fk_list:
                if target == target_table:
                    incoming.append((source_table, fk_col))
        
        return sorted(incoming)
    
    def find_path(self, start_table: str, end_table: str, max_depth: int = 4) -> List[List[str]]:
        """
        Find possible join paths between two tables using breadth-first search
        Returns a list of paths, where each path is a list of table names
        """
        if start_table not in self.tables or end_table not in self.tables:
            return []
        
        # BFS to find all paths
        queue = [(start_table, [start_table])]
        all_paths = []
        visited_states = set()
        
        while queue:
            current_table, path = queue.pop(0)
            
            # Avoid revisiting the same state
            state = (current_table, tuple(path))
            if state in visited_states:
                continue
            visited_states.add(state)
            
            # Check if we've reached the destination
            if current_table == end_table:
                all_paths.append(path)
                continue
            
            # Don't go too deep
            if len(path) >= max_depth:
                continue
            
            # Explore outgoing relationships
            if current_table in self.relationships:
                for fk_col, target_table, _ in self.relationships[current_table]:
                    if target_table not in path:  # Avoid cycles
                        new_path = path + [target_table]
                        queue.append((target_table, new_path))
            
            # Explore incoming relationships
            incoming = self._find_incoming_relationships(current_table)
            for source_table, _ in incoming:
                if source_table not in path:
                    new_path = path + [source_table]
                    queue.append((source_table, new_path))
        
        return all_paths

    def _get_join_info(self, table1: str, table2: str) -> str:
        """Determine the join condition between two tables"""
        # Check if table1 has FK to table2
        if table1 in self.relationships:
            for fk_col, target, _ in self.relationships[table1]:
                if target == table2:
                    return f"JOIN via {table1}.{fk_col} = {table2}.Id"
        
        # Check if table2 has FK to table1
        if table2 in self.relationships:
            for fk_col, target, _ in self.relationships[table2]:
                if target == table1:
                    return f"JOIN via {table2}.{fk_col} = {table1}.Id"
        
        return "⚠️  Join condition unclear"

    # ---------- Enrichment utilities ----------
    def _enrich_relationships_from_templates(self):
        """Parse saved query templates to infer join relationships."""
        if not self.template_dir.exists():
            return
        sql_files = list(self.template_dir.glob("*.sql"))
        for path in sql_files:
            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                continue

            # Look for JOIN <table> ON <left> = <right>
            for match in re.finditer(r"JOIN\s+`?([a-zA-Z0-9_]+)`?\s+ON\s+([^\n;]+)", text, flags=re.IGNORECASE):
                join_table = match.group(1).lower()
                on_clause = match.group(2)

                # try to capture patterns like a.FkCol = b.Id
                cond_match = re.search(r"`?([a-zA-Z0-9_]+)`?\.`?([a-zA-Z0-9_]+)`?\s*=\s*`?([a-zA-Z0-9_]+)`?\.`?([a-zA-Z0-9_]+)`?", on_clause)
                if not cond_match:
                    continue
                left_table, left_col, right_table, right_col = [g.lower() for g in cond_match.groups()]

                # Add inferred relationship if tables exist in schema
                if left_table in self.tables and right_table in self.tables:
                    self._add_relationship(left_table, left_col, right_table, source="template")
                    self._add_relationship(right_table, right_col, left_table, source="template_reverse")

    def _enrich_relationships_from_csv(self, sample_rows: int = 500):
        """Infer lookup relationships from CSVs in the data directory by name/values."""
        if not self.data_dir.exists():
            return

        # Map table -> set of IDs from CSV (sample)
        id_cache: Dict[str, Set[str]] = {}

        def load_ids(table: str) -> Set[str]:
            if table in id_cache:
                return id_cache[table]
            csv_path = self.data_dir / f"{table}.csv"
            ids: Set[str] = set()
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', usecols=['Id'], nrows=sample_rows)
                    ids = set(str(v).strip() for v in df['Id'].dropna().unique())
                except Exception:
                    ids = set()
            id_cache[table] = ids
            return ids

        for table in self.tables:
            csv_path = self.data_dir / f"{table}.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', nrows=sample_rows)
            except Exception:
                continue

            for col in df.columns:
                if not col.endswith('Id') or col == 'Id':
                    continue
                target_table = col[:-2].lower()
                if target_table not in self.tables:
                    continue
                target_ids = load_ids(target_table)
                if not target_ids:
                    continue
                # Overlap check
                col_values = set(str(v).strip() for v in df[col].dropna().unique())
                if not col_values:
                    continue
                overlap = len(col_values & target_ids)
                if overlap > 0:
                    self._add_relationship(table, col, target_table, source="csv")

    def _add_relationship(self, table: str, fk_col: str, target_table: str, source: str = "derived"):
        """Add a relationship edge if not already present."""
        if table not in self.tables or target_table not in self.tables:
            return
        existing = self.relationships.get(table, [])
        if any(fc == fk_col and tgt == target_table for fc, tgt, _ in existing):
            return
        edge = (fk_col, target_table, f"{source}")
        self.relationships.setdefault(table, []).append(edge)

    def enable_enrichment_runtime(self):
        """Enable enrichment after initialization."""
        if self.enrichment_enabled:
            self._print("Enrichment already enabled.")
            return
        self.enrichment_enabled = True
        self._print("Running enrichment from templates and CSV samples...")
        self._enrich_relationships_from_templates()
        self._enrich_relationships_from_csv()
        self._print("Enrichment complete.")

    def suggest_join_path(self, start_table: str, end_table: str) -> None:
        """Find and display the shortest join path between two tables"""

        # Start capturing output to a file
        if self.output_enabled and hasattr(self, 'output'):
            self.output.start_capture('path', start_table, end_table)

        try:
            self._print(f"\n🔍 Finding join path from '{start_table}' to '{end_table}'...\n")

            paths = self.find_path(start_table, end_table, max_depth=5)

            if not paths:
                self._print(f"❌ No path found between {start_table} and {end_table}")
                self._print(f"   (within depth limit of 5 joins)")
                return

            paths.sort(key=len)

            self._print(f"✅ Found {len(paths)} possible path(s):\n")

            for i, path in enumerate(paths[:3], 1):
                self._print(f"{'='*100}")
                self._print(f"PATH {i} ({len(path)-1} join(s)):")
                self._print(f"{'='*100}")

                for j in range(len(path) - 1):
                    current = path[j]
                    next_table = path[j + 1]

                    join_info = self._get_join_info(current, next_table)
                    self._print(f"\n  {current}")
                    self._print(f"    {'↓' if join_info else '↑'} {join_info}")

                self._print(f"\n  {path[-1]}")

                self._print(f"\n📝 SQL Snippet:")
                self._generate_sql_snippet(path)
                self._print()

        finally:
            if self.output_enabled and hasattr(self, 'output'):
                self.output.stop_capture()
    

    
    def _generate_sql_snippet(self, path: List[str]) -> None:
        """Generate a SQL snippet for the join path"""
        if len(path) < 2:
            return
        
        # Create table aliases
        aliases = {table: self._create_alias(table) for table in path}
        
        # Start with FROM clause
        sql = f"FROM `{path[0]}` AS {aliases[path[0]]}\n"
        
        # Add JOIN clauses
        for i in range(len(path) - 1):
            current = path[i]
            next_table = path[i + 1]
            current_alias = aliases[current]
            next_alias = aliases[next_table]
            
            # Determine join condition
            join_type = "JOIN"  # Could be refined to LEFT/INNER based on use case
            
            # Check if current has FK to next
            if current in self.relationships:
                for fk_col, target, _ in self.relationships[current]:
                    if target == next_table:
                        sql += f"{join_type} `{next_table}` AS {next_alias}\n"
                        sql += f"    ON {next_alias}.Id = {current_alias}.{fk_col}\n"
                        break
            else:
                # Check reverse direction
                if next_table in self.relationships:
                    for fk_col, target, _ in self.relationships[next_table]:
                        if target == current:
                            sql += f"{join_type} `{next_table}` AS {next_alias}\n"
                            sql += f"    ON {next_alias}.{fk_col} = {current_alias}.Id\n"
                            break
        
        print(sql)
    
    def _create_alias(self, table_name: str) -> str:
        """Create a short alias for a table name"""
        # Remove common suffixes and take abbreviation
        name = table_name.replace('locale', '').replace('type', '')
        
        # Simple heuristic: use first letters of words or first 3 chars
        if len(name) <= 4:
            return name
        
        # Try to extract capitals or first letters
        words = []
        current_word = name[0]
        for char in name[1:]:
            if char.isupper():
                words.append(current_word)
                current_word = char
            else:
                current_word += char
        words.append(current_word)
        
        if len(words) > 1:
            return ''.join([w[0] for w in words]).lower()
        else:
            return name[:3].lower()
    
    def list_core_entities(self) -> None:
        """List the main business entities (tables without 'locale', 'log', etc.)"""
        print("\n📦 CORE BUSINESS ENTITIES:")
        print(f"{'-'*80}")
        
        # Filter out auxiliary tables
        core_tables = [
            t for t in sorted(self.tables)
            if not any(suffix in t.lower() for suffix in 
                      ['locale', 'log', 'requestlog', 'history', 'import'])
        ]
        
        # Group by domain
        domains = {
            'Account Management': ['account', 'accountholder', 'ecoidentifier'],
            'Vessel Management': ['vessel', 'vesselecoidentifier', 'vesselaccountholder'],
            'Transactions': ['transaction', 'gostransaction', 'transactiontariff'],
            'Service Providers (GOS)': ['gos', 'gosfacility'],
            'Waste Management': ['wasteregistration', 'wastecollection', 'wastetype'],
            'Master Data': ['country', 'vesseltype', 'fueltype', 'hulltype'],
            'User Management': ['appuser', 'approle', 'device']
        }
        
        for domain, keywords in domains.items():
            matching = [t for t in core_tables if any(kw in t.lower() for kw in keywords)]
            if matching:
                print(f"\n{domain}:")
                for table in matching:
                    print(f"  • {table}")
    


    def analyze_entity_relationships(self, entity: str) -> None:
    
        if self.output_enabled and hasattr(self, 'output'):
            self.output.start_capture('analyze', entity)

        try:
            if entity not in self.tables:
                self._print(f"❌ Table '{entity}' not found")
                return

            self._print(f"\n{'='*100}")
            self._print(f"RELATIONSHIP ANALYSIS: {entity}")
            self._print(f"{'='*100}")

            # Direct relationships
            self._print(f"\n🔗 DIRECT RELATIONSHIPS (1 hop):")

            if entity in self.relationships:
                self._print(f"\n  Outgoing (this table references others):")
                for fk_col, target, dtype in self.relationships[entity]:
                    self._print(f"    {entity}.{fk_col} → {target}.Id")

            incoming = self._find_incoming_relationships(entity)
            if incoming:
                self._print(f"\n  Incoming (other tables reference this one):")
                for from_table, via_col in incoming:
                    self._print(f"    {from_table}.{via_col} → {entity}.Id")

            # Two-hop relationships
            self._print(f"\n🔗 TWO-HOP RELATIONSHIPS (common join patterns):")

            one_hop = set()
            if entity in self.relationships:
                one_hop.update([target for _, target, _ in self.relationships[entity]])
            one_hop.update([source for source, _ in incoming])

            two_hop = {}
            for intermediate in one_hop:
                if intermediate in self.relationships:
                    for fk_col, target, _ in self.relationships[intermediate]:
                        if target != entity and target not in one_hop:
                            if intermediate not in two_hop:
                                two_hop[intermediate] = []
                            two_hop[intermediate].append(target)

            if two_hop:
                for intermediate, targets in sorted(two_hop.items()):
                    self._print(f"\n  Via {intermediate}:")
                    for target in targets:
                        self._print(f"    {entity} → {intermediate} → {target}")

        finally:
            if self.output_enabled and hasattr(self, 'output'):
                self.output.stop_capture()
            
    def list_all_tables(self, category=None):
        """List all tables, optionally filtered by category"""

        if self.output_enabled and hasattr(self, 'output'):
            # For list commands, include the category in the filename if provided
            if category:
                self.output.start_capture('list', category)
            else:
                self.output.start_capture('list', 'all')

        try:
            categories = {
                'core': {
                    'keywords': ['account', 'vessel', 'transaction', 'ecoidentifier', 
                                'gos', 'waste', 'facility'],
                    'description': 'Core operational entities'
                },
                'master': {
                    'keywords': ['country', 'vesseltype', 'fueltype', 'wastetype', 
                                'hulltype', 'unitofmeasurement'],
                    'description': 'Master data and classifications'
                },
                'locale': {
                    'keywords': ['locale'],
                    'description': 'Translation and localization tables'
                },
                'link': {
                    'keywords': ['vesselecoidentifier', 'vesselaccountholder'],
                    'description': 'Many-to-many relationship tables'
                },
                'audit': {
                    'keywords': ['audit', 'log', 'history', 'requestlog'],
                    'description': 'Audit trail and logging tables'
                },
                'user': {
                    'keywords': ['user', 'role', 'device', 'token'],
                    'description': 'User management and authentication'
                }
            }

            if category and category.lower() in categories:
                cat_info = categories[category.lower()]
                keywords = cat_info['keywords']
                tables = [t for t in sorted(self.tables) 
                         if any(kw in t.lower() for kw in keywords)]

                self._print(f"\n{'='*100}")
                self._print(f"{category.upper()} TABLES - {cat_info['description']}")
                self._print(f"{'='*100}")
                self._print(f"Found {len(tables)} tables in this category:\n")

            elif category and category.lower() == 'all':
                tables = sorted(self.tables)
                self._print(f"\n{'='*100}")
                self._print(f"ALL TABLES")
                self._print(f"{'='*100}")
                self._print(f"Total: {len(tables)} tables\n")

            elif category:
                self._print(f"\n❌ Unknown category: '{category}'")
                self._print(f"\nAvailable categories:")
                for cat_name, cat_info in sorted(categories.items()):
                    self._print(f"  • {cat_name:10} - {cat_info['description']}")
                self._print(f"  • all        - Show all tables")
                return

            else:
                tables = sorted(self.tables)
                self._print(f"\n{'='*100}")
                self._print(f"ALL TABLES")
                self._print(f"{'='*100}")
                self._print(f"Total: {len(tables)} tables\n")

            for i, table in enumerate(tables, 1):
                self._print(f"  {i:3}. {table}")

            if not category or len(tables) > 10:
                self._print(f"\n💡 Tip: Use 'explore <table_name>' to see details about any table")

        finally:
            if self.output_enabled and hasattr(self, 'output'):
                self.output.stop_capture()

    


def main():
    """Interactive CLI for schema exploration"""
    import sys
    
    schema_path = './AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx'
    
    # Flags
    enable_output = '--no-save' not in sys.argv
    enable_enrichment = '--enrich' in sys.argv
    template_dir = 'saved_queries'
    data_dir = 'data'
    # Clean handled flags
    if '--no-save' in sys.argv:
        sys.argv.remove('--no-save')
    if '--enrich' in sys.argv:
        sys.argv.remove('--enrich')
    
    try:
        explorer = SchemaExplorer(
            schema_path,
            enable_file_output=enable_output,
            template_dir=template_dir,
            data_dir=data_dir,
            enable_enrichment=enable_enrichment
        )
    except Exception as e:
        print(f"❌ Error loading schema: {e}")
        return
    
    print("="*100)
    print("SPE-CDNI SCHEMA EXPLORER")
    print("="*100)
    print("\nAvailable commands:")
    print("  explore <table_name>         - Show detailed info about a table")
    print("  path <table1> <table2>       - Find join path between two tables")
    print("  list [category]              - List all tables (optional: core/master/locale/link)")
    print("  analyze <table_name>         - Comprehensive relationship analysis")
    print("  enrich                       - Enable template/CSV enrichment now")
    print("  quit                         - Exit")
    if enable_output:
        print("\n📄 File output enabled - results will be saved to schema_analysis_output/")
        print("   Use --no-save flag to disable file saving")
    if enable_enrichment:
        print("🔎 Enrichment enabled (templates + CSV sampling).")
    else:
        print("   Use --enrich to infer relationships from saved_queries/ and data/ CSV samples.")
        print("   Or type 'enrich' at the prompt to enable it now.")
    print()
    
    # Rest of the main function remains the same...
    # (Keep all your existing command handling code here)
    
    # If command line args provided, run non-interactively
    # This allows the tool to be used in scripts or one-off commands
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'explore' and len(sys.argv) > 2:
            explorer.explore_table(sys.argv[2])
        
        elif command == 'path' and len(sys.argv) > 3:
            explorer.suggest_join_path(sys.argv[2], sys.argv[3])
        
        elif command == 'list':
            # Handle optional category parameter
            # If user provides a category (e.g., "list core"), use it
            # Otherwise, pass None to show all tables
            category = sys.argv[2] if len(sys.argv) > 2 else None
            explorer.list_all_tables(category)
        
        elif command == 'analyze' and len(sys.argv) > 2:
            explorer.analyze_entity_relationships(sys.argv[2])
        
        elif command == 'enrich':
            explorer.enable_enrichment_runtime()
        
        return
    
    # Interactive mode - keeps running until user quits
    # This is the loop that powers the interactive command prompt
    while True:
        try:
            user_input = input("\n> ").strip()
            
            # Skip empty inputs rather than showing an error
            if not user_input:
                continue
            
            # Split the input into words for command parsing
            parts = user_input.split()
            command = parts[0].lower()
            
            # Handle quit commands - multiple variations for user convenience
            if command in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Explore command - requires a table name as second parameter
            elif command == 'explore' and len(parts) > 1:
                explorer.explore_table(parts[1])
            
            # Path command - requires two table names to find route between them
            elif command == 'path' and len(parts) > 2:
                explorer.suggest_join_path(parts[1], parts[2])
            
            # List command - this is your new addition
            # It works with or without a category parameter
            elif command == 'list':
                # Check if user provided a category as the second word
                # For example: "list core" or "list master"
                category = parts[1] if len(parts) > 1 else None
                explorer.list_all_tables(category)
            
            # Analyze command - requires a table name for relationship analysis
            elif command == 'analyze' and len(parts) > 1:
                explorer.analyze_entity_relationships(parts[1])
            
            elif command == 'enrich':
                explorer.enable_enrichment_runtime()
            
            # Handle invalid commands gracefully with a helpful message
            else:
                print("❌ Invalid command. Try: explore, path, list, analyze, enrich, quit.")
        
        # Allow users to exit with Ctrl+C without seeing a scary error message
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        # Catch any unexpected errors and show them without crashing the tool
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
