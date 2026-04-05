import json
import pickle
from pathlib import Path

# Try to load behaviors from jailbreakbench's cached data
try:
    # jailbreakbench caches downloaded data
    import jailbreakbench.data
    behaviors_path = Path(jailbreakbench.data.__file__).parent / "behaviors.json"
    if behaviors_path.exists():
        with open(behaviors_path) as f:
            behaviors = json.load(f)
            with open('data/alignment/refusal_500.jsonl', 'w') as out:
                for b in behaviors:
                    goal = b.get('goal', b.get('Goal', ''))
                    if goal:
                        out.write(json.dumps({'source': 'jailbreakbench', 'text': goal, 'label': 'refusal'}) + '\n')
            print(f'Wrote {len(behaviors)} behaviors')
    else:
        print(f"Behaviors file not found at {behaviors_path}")
except Exception as e:
    print(f"Error: {e}")