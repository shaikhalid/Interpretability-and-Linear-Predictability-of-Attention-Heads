import json
import sys

if len(sys.argv) < 2:
    print("Usage: python verify_heads.py <ref_target_heads.json>")
    sys.exit(1)

json_path = sys.argv[1]

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

ref_heads_set = set()
target_heads_set = set()

for group in data.values():
    # Add all reference heads as tuples
    for ref_head in group.get('ref_heads', []):
        ref_heads_set.add(tuple(ref_head))
    # Add all target heads as tuples
    for target_head in group.get('target_heads', []):
        target_heads_set.add(tuple(target_head))

# Find any heads that are both reference and target
violations = ref_heads_set & target_heads_set

if violations:
    print("Violation: The following heads are both reference and target heads:")
    for head in violations:
        print(head)
else:
    print("Success: No reference head is ever a target head.")