"""
Query Generator
Orchestrates SQL query construction from user intent and schema information
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class QueryState:
    """Track the state of query construction"""
    base_table: str = ""
    joined_tables: List[Tuple[str, str]] = field(default_factory=list)  # (table, join_condition)
    select_columns: List[str] = field(default_factory=list)
    where_conditions: List[str] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)
    order_by_columns: List[str] = field(default_factory=list)
    having_conditions: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    table_aliases: Dict[str, str] = field(default_factory=dict)


class QueryGenerator:
    """
    Generates MySQL-compatible SQL queries from structured requirements.
    MVP version - supports SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY
    """

    def __init__(self):
        self.state = QueryState()

    def reset(self):
        """Reset the query state for a new query"""
        self.state = QueryState()

    def set_base_table(self, table_name: str):
        """Set the primary table for the query"""
        self.state.base_table = table_name
        alias = self._create_alias(table_name)
        self.state.table_aliases[table_name] = alias

    def add_join(self, table_name: str, join_condition: str, join_type: str = "JOIN"):
        """
        Add a table join to the query.

        Args:
            table_name: Table to join
            join_condition: Join condition (e.g., "t.VesselId = v.Id")
            join_type: Type of join (JOIN, LEFT JOIN, etc.)
        """
        if table_name not in self.state.table_aliases:
            alias = self._create_alias(table_name)
            self.state.table_aliases[table_name] = alias

        self.state.joined_tables.append((table_name, f"{join_type} `{table_name}` AS {self.state.table_aliases[table_name]} ON {join_condition}"))

    def add_select_column(self, column_expr: str, alias: Optional[str] = None):
        """
        Add a column to the SELECT clause.

        Args:
            column_expr: Column expression (can include functions)
            alias: Optional alias for the column
        """
        if alias:
            self.state.select_columns.append(f"{column_expr} AS {alias}")
        else:
            self.state.select_columns.append(column_expr)

    def add_where_condition(self, condition: str):
        """Add a WHERE condition"""
        self.state.where_conditions.append(condition)

    def add_group_by(self, column_expr: str):
        """Add a GROUP BY column"""
        self.state.group_by_columns.append(column_expr)

    def add_order_by(self, column_expr: str, direction: str = "ASC"):
        """Add an ORDER BY column"""
        self.state.order_by_columns.append(f"{column_expr} {direction}")

    def add_having_condition(self, condition: str):
        """Add a HAVING condition"""
        self.state.having_conditions.append(condition)

    def set_limit(self, limit: int):
        """Set LIMIT clause"""
        self.state.limit = limit

    def build_join_chain_from_path(self, path: List[str], join_info: List[Dict]):
        """
        Build joins from a path of tables.

        Args:
            path: List of table names in order
            join_info: List of join dictionaries with conditions
        """
        if not path:
            return

        # Pre-create aliases for all tables in the path so conditions can be alias-replaced correctly
        for table in path:
            if table not in self.state.table_aliases:
                self.state.table_aliases[table] = self._create_alias(table)

        # Set base table (keeps existing alias)
        self.state.base_table = path[0]

        # Add joins
        for join in join_info:
            from_table = join['from_table']
            to_table = join['to_table']
            condition = join['condition']
            join_type = join.get('type', 'JOIN')

            # Replace table names with aliases in condition
            condition_with_aliases = self._apply_aliases_to_condition(condition)

            self.add_join(to_table, condition_with_aliases, join_type)

    def _apply_aliases_to_condition(self, condition: str) -> str:
        """Replace table names in condition with their aliases"""
        result = condition
        for table, alias in self.state.table_aliases.items():
            result = result.replace(f"{table}.", f"{alias}.")
        return result

    def _create_alias(self, table_name: str) -> str:
        """
        Create a short alias for a table name.

        Args:
            table_name: Table name

        Returns:
            Alias string
        """
        # Remove common suffixes
        name = table_name.lower().replace('locale', '').replace('type', '')

        # Try to extract capitals or use first letters
        if len(name) <= 3:
            return name

        # Extract capitals if present
        capitals = ''.join([c for c in name if c.isupper()])
        if len(capitals) >= 2:
            return capitals[:3].lower()

        # Use first 2-3 characters
        return name[:2] if len(name) > 3 else name

    def generate_sql(self) -> str:
        """
        Generate the complete SQL query.

        Returns:
            Formatted SQL string
        """
        if not self.state.base_table:
            raise ValueError("No base table set. Cannot generate query.")

        if not self.state.select_columns:
            raise ValueError("No columns selected. Cannot generate query.")

        sql_parts = []

        # SELECT clause
        sql_parts.append("SELECT")
        sql_parts.append("    " + ",\n    ".join(self.state.select_columns))

        # FROM clause
        base_alias = self.state.table_aliases[self.state.base_table]
        sql_parts.append(f"FROM `{self.state.base_table}` AS {base_alias}")

        # JOIN clauses
        for _, join_clause in self.state.joined_tables:
            sql_parts.append(join_clause)

        # WHERE clause
        if self.state.where_conditions:
            sql_parts.append("WHERE " + self._format_conditions(self.state.where_conditions, "AND"))

        # GROUP BY clause
        if self.state.group_by_columns:
            sql_parts.append("GROUP BY " + ", ".join(self.state.group_by_columns))

        # HAVING clause
        if self.state.having_conditions:
            sql_parts.append("HAVING " + self._format_conditions(self.state.having_conditions, "AND"))

        # ORDER BY clause
        if self.state.order_by_columns:
            sql_parts.append("ORDER BY " + ", ".join(self.state.order_by_columns))

        # LIMIT clause
        if self.state.limit:
            sql_parts.append(f"LIMIT {self.state.limit}")

        return "\n".join(sql_parts) + ";"

    def _format_conditions(self, conditions: List[str], operator: str) -> str:
        """Format multiple conditions with proper indentation"""
        if len(conditions) == 1:
            return conditions[0]

        formatted = conditions[0]
        for cond in conditions[1:]:
            formatted += f"\n  {operator} {cond}"
        return formatted

    def generate_with_comments(self, description: str = "") -> str:
        """
        Generate SQL with explanatory comments.

        Args:
            description: Optional description of query purpose

        Returns:
            SQL with comments
        """
        sql = self.generate_sql()

        comments = []
        if description:
            comments.append(f"-- {description}")
            comments.append("--")

        # Add metadata comments
        comments.append(f"-- Base table: {self.state.base_table}")
        if self.state.joined_tables:
            comments.append(f"-- Joined tables: {', '.join(t for t, _ in self.state.joined_tables)}")

        if self.state.where_conditions:
            comments.append(f"-- Filters: {len(self.state.where_conditions)} condition(s)")

        if self.state.group_by_columns:
            comments.append(f"-- Grouped by: {', '.join(self.state.group_by_columns)}")

        comments.append("--")
        comments.append("")

        return "\n".join(comments) + sql

    def explain_query(self) -> str:
        """
        Generate a human-readable explanation of the query.

        Returns:
            Explanation string
        """
        explanation = []

        explanation.append("Query Explanation:")
        explanation.append("=" * 50)

        # What data source
        explanation.append(f"\nData Source: {self.state.base_table}")

        # What joins
        if self.state.joined_tables:
            explanation.append(f"\nJoins:")
            for table, _ in self.state.joined_tables:
                explanation.append(f"  - {table}")

        # What we're selecting
        explanation.append(f"\nColumns Selected:")
        for col in self.state.select_columns:
            explanation.append(f"  - {col}")

        # What filters
        if self.state.where_conditions:
            explanation.append(f"\nFilters Applied:")
            for cond in self.state.where_conditions:
                explanation.append(f"  - {cond}")

        # What grouping
        if self.state.group_by_columns:
            explanation.append(f"\nGrouped By:")
            for col in self.state.group_by_columns:
                explanation.append(f"  - {col}")

        # What ordering
        if self.state.order_by_columns:
            explanation.append(f"\nOrdered By:")
            for col in self.state.order_by_columns:
                explanation.append(f"  - {col}")

        return "\n".join(explanation)

    def get_estimated_complexity(self) -> str:
        """
        Estimate query complexity.

        Returns:
            Complexity rating: "simple", "moderate", "complex"
        """
        complexity_score = 0

        # Base complexity from joins
        complexity_score += len(self.state.joined_tables)

        # Add complexity for aggregations
        if self.state.group_by_columns:
            complexity_score += 2

        # Add complexity for multiple conditions
        complexity_score += len(self.state.where_conditions) // 3

        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "moderate"
        else:
            return "complex"


def create_simple_select(table: str, columns: List[str], filters: Optional[Dict[str, str]] = None) -> str:
    """
    Helper function to quickly create a simple SELECT query.

    Args:
        table: Table name
        columns: List of column names
        filters: Optional dictionary of column: value filters

    Returns:
        SQL query string
    """
    generator = QueryGenerator()
    generator.set_base_table(table)

    alias = generator.state.table_aliases[table]

    for col in columns:
        generator.add_select_column(f"{alias}.{col}")

    if filters:
        for col, value in filters.items():
            if isinstance(value, str):
                generator.add_where_condition(f"{alias}.{col} = '{value}'")
            else:
                generator.add_where_condition(f"{alias}.{col} = {value}")

    return generator.generate_sql()
