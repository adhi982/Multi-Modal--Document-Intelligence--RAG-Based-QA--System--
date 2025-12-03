"""Debug chunk structure."""
import json

with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print("Sample chunk structure:")
print(json.dumps(chunks[0], indent=2))

print("\n" + "=" * 80)
print(f"\nChecking page_num field in {len(chunks)} chunks:")
has_page = sum(1 for c in chunks if c.get('page_num') is not None)
print(f"  Chunks with page_num: {has_page}/{len(chunks)}")

# Check alternative field names
alt_fields = ['page', 'page_number', 'metadata']
for field in alt_fields:
    count = sum(1 for c in chunks if c.get(field) is not None)
    if count > 0:
        print(f"  Chunks with '{field}': {count}")
