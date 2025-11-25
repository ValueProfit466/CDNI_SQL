"""
Intent Interpreter
Parses natural language input to identify entities, intent, and query requirements
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import re
from utils.data_catalog import DataCatalog


@dataclass
class ParsedIntent:
    """Structured representation of user's query intent"""
    entities: List[str]  # Tables/domains mentioned (vessel, transaction, waste, etc.)
    intent_type: str  # aggregation, listing, filtering, comparison, trend
    aggregation: Optional[str] = None  # SUM, COUNT, AVG, etc.
    time_dimension: Optional[str] = None  # year, month, quarter
    filters: Dict[str, str] = None  # Detected filter requirements
    clarifications: List[str] = None  # Questions to ask user
    original_question: str = ""
    confidence: float = 0.0  # 0-1 confidence score

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.clarifications is None:
            self.clarifications = []


class IntentInterpreter:
    """
    Interprets natural language questions about the database.
    Uses keyword matching and pattern recognition (not complex NLP).
    Enhanced with relationship metadata awareness.
    """

    def __init__(self, schema_navigator=None, data_catalog: Optional[DataCatalog] = None):
        """
        Initialize intent interpreter.

        Args:
            schema_navigator: Optional SchemaNavigator instance for enhanced relationship metadata
        """
        self.navigator = schema_navigator  # Access to enhanced metadata
        self.data_catalog = data_catalog  # Access to CSV-backed lookup values

        # Entity keywords - maps business concepts to database tables
        self.entity_keywords = {
            'vessel': {
                'keywords': ['vessel', 'schepen', 'ship', 'boat', 'ENI', 'vaartuig'],
                'primary_table': 'vessel',
                'related_tables': ['vesselecoidentifier', 'vesselaccountholder', 'vesseltype']
            },
            'transaction': {
                'keywords': ['transaction', 'transactie', 'payment', 'betaling', 'charge'],
                'primary_table': 'transaction',
                'related_tables': ['gostransaction', 'transactiontariff']
            },
            'waste': {
                'keywords': ['waste', 'afval', 'verwijdering', 'disposal', 'garbage', 'collection'],
                'primary_table': 'wasteregistration',
                'related_tables': ['wasteregistrationdetail', 'wastetype', 'wastecollectioncompany']
            },
            'account': {
                'keywords': ['account', 'rekening', 'eco-id', 'ecoidentifier', 'card', 'kaart'],
                'primary_table': 'account',
                'related_tables': ['accountholder', 'ecoidentifier']
            },
            'country': {
                'keywords': ['country', 'land', 'nation', 'geographic', 'geografisch',
                            'Belgium', 'België', 'Netherlands', 'Nederland', 'France', 'Frankrijk'],
                'primary_table': 'country',
                'related_tables': []
            },
            'facility': {
                'keywords': ['facility', 'faciliteit', 'port', 'haven', 'station', 'GOS', 'terminal'],
                'primary_table': 'gosfacility',
                'related_tables': ['gos', 'wastecollectionfacility']
            },
            'fuel': {
                'keywords': ['fuel', 'brandstof', 'bunker', 'bunkeren', 'fueltype'],
                'primary_table': 'fueltype',
                'related_tables': []
            }
        }

        # Intent patterns - what kind of query is this?
        self.intent_patterns = {
            'aggregation': {
                'keywords': ['total', 'totaal', 'sum', 'count', 'aantal', 'average', 'gemiddelde',
                            'per', 'by', 'door'],
                'requires_grouping': True
            },
            'filtering': {
                'keywords': ['where', 'for', 'voor', 'in', 'from', 'van', 'by', 'with'],
                'requires_grouping': False
            },
            'listing': {
                'keywords': ['list', 'lijst', 'show', 'toon', 'display', 'weergeven', 'all', 'alle', 'which', 'welke'],
                'requires_grouping': False
            },
            'comparison': {
                'keywords': ['compare', 'vergelijk', 'versus', 'vs', 'difference', 'verschil'],
                'requires_grouping': True
            },
            'trend': {
                'keywords': ['over time', 'trend', 'yearly', 'jaarlijks', 'monthly', 'maandelijks',
                            'growth', 'groei', 'evolution', 'evolutie'],
                'requires_grouping': True
            }
        }

        # Aggregation keywords
        self.aggregation_keywords = {
            'SUM': ['total', 'totaal', 'sum', 'somme'],
            'COUNT': ['count', 'aantal', 'number', 'how many', 'hoeveel'],
            'AVG': ['average', 'gemiddelde', 'mean', 'avg'],
            'MAX': ['maximum', 'max', 'highest', 'hoogste'],
            'MIN': ['minimum', 'min', 'lowest', 'laagste']
        }

        # Time dimension keywords
        self.time_keywords = {
            'year': ['year', 'jaar', 'yearly', 'jaarlijks', 'annual', 'annuel'],
            'month': ['month', 'maand', 'monthly', 'maandelijks'],
            'quarter': ['quarter', 'kwartaal', 'quarterly', 'kwartaallijks'],
            'day': ['day', 'dag', 'daily', 'dagelijks']
        }

        # Specific country names for filter detection (augmented by catalog if present)
        self.countries = ['Belgium', 'België', 'Netherlands', 'Nederland', 'France', 'Frankrijk',
                         'Germany', 'Duitsland', 'Luxembourg', 'Luxemburg', 'Switzerland', 'Zwitserland']

        # Transaction status code intent mapping
        self.status_synonyms = {
            'success': ['processed', 'approved', 'booked', 'good', 'successful', 'valid'],
            'pending': ['pending', 'waiting', 'withheld', 'initiated', 'in progress'],
            'failed': ['failed', 'declined', 'denied', 'error', 'rejected'],
        }
        self.status_code_map = {
            'success': ['B'],
            'pending': ['I', 'P', 'W'],
            'failed': ['F', 'D'],
        }

    def parse_question(self, question: str) -> ParsedIntent:
        """
        Main entry point - parse a natural language question.

        Args:
            question: User's natural language question

        Returns:
            ParsedIntent object with structured interpretation
        """
        question_lower = question.lower()

        # Identify entities
        entities = self._identify_entities(question_lower)

        # Detect intent type
        intent_type = self._detect_intent_type(question_lower)

        # Detect aggregation if present
        aggregation = self._detect_aggregation(question_lower)

        # Detect time dimension
        time_dimension = self._detect_time_dimension(question_lower)

        # Extract filters
        filters = self._extract_filters(question_lower)

        # Generate clarifying questions
        clarifications = self._generate_clarifications(entities, filters, question_lower)

        # Calculate confidence score
        confidence = self._calculate_confidence(entities, intent_type)

        return ParsedIntent(
            entities=entities,
            intent_type=intent_type,
            aggregation=aggregation,
            time_dimension=time_dimension,
            filters=filters,
            clarifications=clarifications,
            original_question=question,
            confidence=confidence
        )

    def _identify_entities(self, question: str) -> List[str]:
        """Identify which database entities are mentioned in the question"""
        entities = []

        for entity_name, entity_info in self.entity_keywords.items():
            keywords = [keyword.lower() for keyword in entity_info['keywords']]
            if any(keyword in question for keyword in keywords):
                entities.append(entity_name)

        return entities

    def _detect_intent_type(self, question: str) -> str:
        """Determine the type of query (aggregation, listing, etc.)"""
        intent_scores = {}

        for intent_name, intent_info in self.intent_patterns.items():
            keywords = intent_info['keywords']
            score = sum(1 for keyword in keywords if keyword in question)
            intent_scores[intent_name] = score

        # Return intent with highest score, default to 'listing'
        if not intent_scores or max(intent_scores.values()) == 0:
            return 'listing'

        return max(intent_scores, key=intent_scores.get)

    def _detect_aggregation(self, question: str) -> Optional[str]:
        """Detect which aggregation function is needed"""
        for agg_func, keywords in self.aggregation_keywords.items():
            if any(keyword in question for keyword in keywords):
                return agg_func
        return None

    def _detect_time_dimension(self, question: str) -> Optional[str]:
        """Detect time-based grouping dimension"""
        for dimension, keywords in self.time_keywords.items():
            if any(keyword in question for keyword in keywords):
                return dimension
        return None

    def _extract_filters(self, question: str) -> Dict[str, str]:
        """Extract filter requirements from the question"""
        filters = {}

        # Check for country filters using catalog tokens when available
        country_tokens: Dict[str, str] = {}
        if self.data_catalog:
            for row in self.data_catalog.country_records():
                name = (row.get("Name") or "").strip()
                if not name:
                    continue
                key = name.lower()
                country_tokens[key] = name
                for iso_key in ("ISO2", "ISO3"):
                    iso_val = (row.get(iso_key) or "").strip()
                    if iso_val:
                        country_tokens[iso_val.lower()] = name
        # Fall back to static list if catalog missing
        for country in self.countries:
            country_tokens.setdefault(country.lower(), country)

        matched_country = None
        question_words = question
        # Prefer longest tokens to avoid ISO2 collisions with substrings
        for token, canonical in sorted(country_tokens.items(), key=lambda kv: len(kv[0]), reverse=True):
            if not token:
                continue
            if re.search(rf'\b{re.escape(token)}\b', question_words):
                matched_country = canonical
                break
        if matched_country:
            filters['country'] = matched_country

        # Check for date range mentions
        # Pattern: year mentions like "2023", "2024", "in 2023"
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, question)
        if years:
            if len(years) == 1:
                filters['year'] = years[0]
            elif len(years) == 2:
                filters['year_range'] = f"{years[0]}-{years[1]}"

        # Check for "last X months/years"
        if 'last' in question or 'laatste' in question:
            filters['time_relative'] = 'recent'

        # Check for waste type mentions using catalog
        if self.data_catalog:
            for name in self.data_catalog.waste_type_names():
                if not name:
                    continue
                lower_name = name.lower()
                head = lower_name.split('-')[0].strip() if '-' in lower_name else lower_name
                if lower_name in question or (head and head in question):
                    filters['waste_type'] = name
                    break

        # Transaction status intent
        for intent_label, keywords in self.status_synonyms.items():
            if any(keyword in question for keyword in keywords):
                filters['status_codes'] = self.status_code_map[intent_label]
                break

        return filters

    def _generate_clarifications(self, entities: List[str], filters: Dict[str, str],
                                 question: str) -> List[str]:
        """Generate clarifying questions based on ambiguities"""
        clarifications = []

        # If country mentioned, clarify which aspect
        if 'country' in filters and ('vessel' in entities or 'account' in entities):
            clarifications.append(
                f"When you mention {filters['country']}, do you mean:\n"
                "  (a) Vessels registered in this country?\n"
                "  (b) Account holders based in this country?\n"
                "  (c) Facilities located in this country?"
            )

        # If waste mentioned, clarify what to measure
        if 'waste' in entities and not any(word in question for word in ['quantity', 'hoeveelheid', 'count', 'aantal']):
            clarifications.append(
                "What would you like to know about waste:\n"
                "  (a) Total quantity collected?\n"
                "  (b) Number of collection events?\n"
                "  (c) Types of waste collected?"
            )

        # If transaction/fuel mentioned, clarify type
        if 'transaction' in entities or 'fuel' in entities:
            if not any(word in question for word in ['gos', 'bunker', 'waste', 'afval']):
                clarifications.append(
                    "Which type of transaction:\n"
                    "  (a) GOS fuel transactions (bunkering)?\n"
                    "  (b) All transactions?"
                )

        # If no entities detected, ask for clarification
        if not entities:
            clarifications.append(
                "I couldn't identify the main subject of your query. Are you asking about:\n"
                "  - Vessels/Ships?\n"
                "  - Transactions?\n"
                "  - Waste collection?\n"
                "  - Accounts/ECO-IDs?\n"
                "Please clarify what you'd like to analyze."
            )

        return clarifications

    def _calculate_confidence(self, entities: List[str], intent_type: str) -> float:
        """Calculate confidence score (0-1) for the interpretation"""
        confidence = 0.0

        # Base confidence from entity detection
        if len(entities) > 0:
            confidence += 0.4
        if len(entities) > 1:
            confidence += 0.2

        # Confidence from intent detection
        if intent_type != 'listing':  # More specific than default
            confidence += 0.3
        else:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_primary_tables(self, entities: List[str]) -> List[str]:
        """Get the primary database tables for identified entities"""
        tables = []
        for entity in entities:
            if entity in self.entity_keywords:
                tables.append(self.entity_keywords[entity]['primary_table'])
        return tables

    def get_related_tables(self, entities: List[str]) -> List[str]:
        """Get related tables that might be needed for joins"""
        tables = set()
        for entity in entities:
            if entity in self.entity_keywords:
                tables.update(self.entity_keywords[entity]['related_tables'])
        return list(tables)

    def get_related_tables_with_confidence(self, entities: List[str]) -> Dict[str, Dict]:
        """
        Get related tables with confidence scores from metadata (if available).

        Args:
            entities: List of entity names

        Returns:
            Dictionary mapping entity to table info with confidence:
            {
                'vessel': {
                    'primary': 'vessel',
                    'related': [
                        {'table': 'vesselecoidentifier', 'confidence': 'HIGH'},
                        {'table': 'vesseltype', 'confidence': 'MEDIUM'}
                    ]
                }
            }
        """
        result = {}

        for entity in entities:
            if entity not in self.entity_keywords:
                continue

            entity_info = self.entity_keywords[entity]
            result[entity] = {
                'primary': entity_info['primary_table'],
                'related': []
            }

            # Add confidence info for related tables if navigator available
            for table in entity_info['related_tables']:
                confidence_info = {'table': table, 'confidence': 'MEDIUM'}

                # If we have navigator with metadata, check confidence
                if self.navigator and hasattr(self.navigator, 'has_enhanced_metadata'):
                    if self.navigator.has_enhanced_metadata:
                        # Check if this is a high-confidence relationship
                        if self.navigator.is_junction_table(table):
                            confidence_info['confidence'] = 'HIGH'
                            confidence_info['type'] = 'junction_table'

                result[entity]['related'].append(confidence_info)

        return result

    def suggest_similar_questions(self, intent: ParsedIntent) -> List[str]:
        """Suggest similar questions user might ask based on current intent"""
        suggestions = []

        if 'vessel' in intent.entities and intent.intent_type == 'aggregation':
            suggestions.extend([
                "How many Belgian vessels are registered?",
                "What is the total fuel consumption by vessel type?",
                "Show me transaction counts per vessel per year"
            ])

        if 'waste' in intent.entities:
            suggestions.extend([
                "Total waste collected per year for Belgium",
                "Which waste types are most common?",
                "Show waste collection by vessel type"
            ])

        if 'transaction' in intent.entities or 'fuel' in intent.entities:
            suggestions.extend([
                "Total fuel transactions per country per year",
                "Average transaction amount by fuel type",
                "Geographic distribution of bunkering operations"
            ])

        return suggestions[:3]  # Return top 3 suggestions
