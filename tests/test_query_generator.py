from generators.query_generator import QueryGenerator, create_simple_select


def test_generate_sql_simple_select_with_aliases():
    generator = QueryGenerator()
    generator.set_base_table("vessel")
    alias = generator.state.table_aliases["vessel"]

    generator.add_select_column(f"{alias}.Id")
    generator.add_select_column(f"{alias}.Name")

    sql = generator.generate_sql()

    assert sql == (
        "SELECT\n"
        f"    {alias}.Id,\n"
        f"    {alias}.Name\n"
        f"FROM `vessel` AS {alias};"
    )


def test_create_simple_select_adds_filters_and_aliases():
    sql = create_simple_select(
        table="account",
        columns=["Id", "Name"],
        filters={"CountryId": "BE", "Status": 1},
    )

    assert sql.startswith("SELECT\n    ac.Id,\n    ac.Name\nFROM `account` AS ac")
    assert "WHERE ac.CountryId = 'BE'\n  AND ac.Status = 1" in sql
    assert sql.strip().endswith(";")
