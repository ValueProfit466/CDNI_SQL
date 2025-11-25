from core.intent_interpreter import IntentInterpreter


def test_detects_listing_entities_and_confidence():
    interpreter = IntentInterpreter()

    intent = interpreter.parse_question("List all vessels")

    assert intent.entities == ['vessel']
    assert intent.intent_type == 'listing'
    assert intent.aggregation is None
    assert intent.time_dimension is None
    assert intent.filters == {}
    assert intent.clarifications == []
    assert intent.confidence == 0.5


def test_extracts_filters_aggregation_and_time_dimension():
    interpreter = IntentInterpreter()

    intent = interpreter.parse_question("count waste for Belgium per year")

    assert set(intent.entities) == {'waste', 'country'}
    assert intent.intent_type == 'aggregation'
    assert intent.aggregation == 'COUNT'
    assert intent.time_dimension == 'year'
    assert intent.filters == {'country': 'Belgium'}
    assert intent.clarifications == []
    assert abs(intent.confidence - 0.9) < 1e-6


def test_detects_status_codes_from_keywords():
    interpreter = IntentInterpreter()

    intent = interpreter.parse_question("pending fuel transactions")

    assert intent.filters.get('status_codes') == ['I', 'P', 'W']


class _StubCatalog:
    def country_records(self):
        return []

    def waste_type_names(self):
        return ["Bilge water - Aft engine compartment"]


def test_detects_waste_type_from_catalog():
    interpreter = IntentInterpreter(data_catalog=_StubCatalog())

    intent = interpreter.parse_question("total bilge water collected")

    assert intent.filters.get('waste_type') == "Bilge water - Aft engine compartment"
