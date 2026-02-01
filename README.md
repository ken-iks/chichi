## CHICHI action recognition model (or something)! 

### Run local:
`uv run main.py`

### Run in cloud:
`uv run modal run server.py`

### Check run metadata:
```
python3 -c "
import json
data = json.load(open('output/teams.json'))
print(f'Team 1 ({data[0][\"team_color\"]}): {len(data[0][\"players\"])} players')
print(f'Team 2 ({data[1][\"team_color\"]}): {len(data[1][\"players\"])} players')
"
```
