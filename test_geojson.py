import json

with open('municipalitiescolombia.geojson', encoding='utf-8') as f:
    data = json.load(f)

print(data['features'][0]['properties'])