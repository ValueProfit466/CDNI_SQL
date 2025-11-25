"""
Quick test of the enhanced relationship discovery system
"""

from core.schema_navigator import SchemaNavigator

# Initialize with metadata
navigator = SchemaNavigator(
    excel_path='AlleCDNI_TableNames_Columns_Dtypes_SuperUser_SQLBuilder.xlsx',
    metadata_path='config/relationship_metadata.json'
)

print("="*70)
print("ENHANCED RELATIONSHIP DISCOVERY TEST")
print("="*70)
print(f"\nMetadata loaded: {navigator.has_enhanced_metadata}")
print(f"Total tables: {len(navigator.tables)}")

# Test 1: Get enhanced FKs for 'account' table
print("\n" + "="*70)
print("TEST 1: Enhanced FK Discovery for 'account' table")
print("="*70)

enhanced_fks = navigator.get_foreign_keys_enhanced('account')

for fk_col, target, dtype, meta in enhanced_fks:
    print(f"\n  {fk_col} -> {target}")
    print(f"    Discovery method: {meta['discovery_method']}")
    print(f"    Confidence: {meta['confidence']}")
    if meta.get('match_rate'):
        print(f"    Match rate: {meta['match_rate']:.1%}")
    if meta.get('warnings'):
        for warning in meta['warnings']:
            print(f"    Warning: {warning}")

# Test 2: Get enhanced FKs for 'gostransaction' table (includes RecipientId exception)
print("\n" + "="*70)
print("TEST 2: Enhanced FK Discovery for 'gostransaction' table")
print("="*70)
print("This table includes the problematic 'RecipientId' relationship")
print()

enhanced_fks = navigator.get_foreign_keys_enhanced('gostransaction')

# Show just the first 5 for brevity
for fk_col, target, dtype, meta in enhanced_fks[:5]:
    print(f"\n  {fk_col} -> {target}")
    print(f"    Discovery method: {meta['discovery_method']}")
    print(f"    Confidence: {meta['confidence']}")
    if meta.get('match_rate') is not None:
        print(f"    Match rate: {meta['match_rate']:.1%}")

# Test 3: Check if RecipientId exception works
print("\n" + "="*70)
print("TEST 3: RecipientId Exception Mapping")
print("="*70)

recipient_fk = next(
    (fk for fk in enhanced_fks if fk[0] == 'RecipientId'),
    None
)

if recipient_fk:
    fk_col, target, dtype, meta = recipient_fk
    print(f"\n✓ SUCCESS: RecipientId correctly mapped to '{target}'")
    print(f"  (Without enhancement, convention would suggest 'recipient' table)")
    print(f"  Discovery method: {meta['discovery_method']}")
    print(f"  Confidence: {meta['confidence']}")
else:
    print("\n✗ FAILED: RecipientId not found")

# Test 4: Junction table detection
print("\n" + "="*70)
print("TEST 4: Junction Table Detection")
print("="*70)

junction_tables = ['vesselecoidentifier', 'vesselaccountholder']
for table in junction_tables:
    is_junction = navigator.is_junction_table(table)
    print(f"  {table}: {'✓ Junction table' if is_junction else '✗ Not junction'}")

print("\n" + "="*70)
print("ALL TESTS COMPLETE!")
print("="*70)
