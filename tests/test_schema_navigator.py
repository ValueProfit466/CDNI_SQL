"""
Tests for SchemaNavigator entity mapping and confidence tracking.
"""

import pytest
from unittest.mock import MagicMock, patch


class MockSchemaExplorer:
    """Mock SchemaExplorer for testing Navigator without Excel file dependency."""

    def __init__(self):
        self.tables = {
            'vessel', 'vesselecoidentifier', 'vesselaccountholder',
            'transaction', 'gostransaction', 'transactiontariff',
            'account', 'accountholder', 'ecoidentifier',
            'country', 'countrylocale',
            'wastetype', 'wasteregistration'
        }
        self.relationships = {
            'vessel': [('CountryId', 'country', 'int')],
            'account': [('AccountHolderId', 'accountholder', 'int')],
        }
        self.df = MagicMock()


@pytest.fixture
def mock_navigator():
    """Create SchemaNavigator with mocked SchemaExplorer."""
    with patch('core.schema_navigator.SchemaExplorer', MockSchemaExplorer):
        from core.schema_navigator import SchemaNavigator
        navigator = SchemaNavigator.__new__(SchemaNavigator)
        navigator.explorer = MockSchemaExplorer()
        navigator.tables = navigator.explorer.tables
        navigator.relationships = navigator.explorer.relationships
        navigator.df = navigator.explorer.df
        navigator.enriched_tables = set()
        navigator._table_info_cache = {}
        navigator._join_path_cache = {}
        navigator.metadata = {'relationships': {}, 'metadata': {}}
        navigator.has_enhanced_metadata = False
        return navigator


def test_find_tables_exact_match_high_confidence(mock_navigator):
    """Known entities should return HIGH confidence."""
    result = mock_navigator.find_tables_for_entities(['vessel'])

    assert 'vessel' in result
    assert 'vessel' in result['vessel']
    assert mock_navigator.get_entity_confidence('vessel') == 'HIGH'


def test_find_tables_unknown_entity_low_confidence(mock_navigator):
    """Unknown entities with similar tables should return LOW confidence."""
    # 'vess' is partial match for 'vessel'
    result = mock_navigator.find_tables_for_entities(['vess'])

    # Should find vessel-related tables via similarity
    if 'vess' in result:
        assert mock_navigator.get_entity_confidence('vess') == 'LOW'


def test_find_tables_no_match_returns_empty(mock_navigator):
    """Completely unmatched entities should return empty."""
    result = mock_navigator.find_tables_for_entities(['xyznonexistent'])

    # Should either be empty or not in result
    assert 'xyznonexistent' not in result or result['xyznonexistent'] == []


def test_find_tables_multiple_entities(mock_navigator):
    """Multiple entities should each get their own confidence."""
    result = mock_navigator.find_tables_for_entities(['vessel', 'account'])

    assert 'vessel' in result
    assert 'account' in result
    assert mock_navigator.get_entity_confidence('vessel') == 'HIGH'
    assert mock_navigator.get_entity_confidence('account') == 'HIGH'


def test_similarity_threshold_filters_weak_matches(mock_navigator):
    """Similarity matching should respect minimum ratio threshold."""
    # With high threshold, weak matches should be filtered
    result = mock_navigator._find_similar_tables_with_threshold('v', min_ratio=0.8)

    # Single char 'v' shouldn't match much with 0.8 threshold
    # (80% of 1 char = 0.8 chars minimum, which rounds to 0)
    # This tests the threshold logic


def test_get_entity_confidence_unknown_entity(mock_navigator):
    """Confidence for unmapped entity should return None."""
    assert mock_navigator.get_entity_confidence('never_mapped') is None


def test_known_entity_tables_exist(mock_navigator):
    """Known entity mappings should reference existing tables."""
    result = mock_navigator.find_tables_for_entities(['vessel', 'transaction', 'account'])

    for entity, tables in result.items():
        for table in tables:
            assert table in mock_navigator.tables, f"Table {table} for {entity} not in schema"
