import requests
import json


test_ids = ['A0A151EH88', 'A0A1I4KS07', 'A1AY86', 'F4HT77']

for uniprot_id in test_ids:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"\n{'=' * 60}")
        print(f"UniProt ID: {uniprot_id}")
        print(f"{'=' * 60}")
        print(json.dumps(data, indent=2))