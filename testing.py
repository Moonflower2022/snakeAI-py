import json

# Load the existing JSON file
with open('stable_baselines/6x6models/info.json', 'r') as f:
    data = json.load(f)

# Add new model descriptions
new_model_descriptions = {
    "ppo5_1": {
        "board_size": "5x5",
        "env": "5action",
        "rewards": "-1 for dying, 1 for getting fruit, 100 for winning",
        "gamma": "0.9",
        "ent_coef": "0.02",
        "learning_rate": "0.0008895296207610578",
        "strength": "strong",
        "notes": "dies randomly sometimes"
    },
    "ppo6_1": {
        "board_size": "6x6",
        "env": "6action",
        "rewards": "-1 for dying, 1 for getting fruit, 100 for winning",
        "gamma": "0.95",
        "ent_coef": "0.02",
        "learning_rate": "0.0008895296207610578",
        "strength": "strong",
        "notes": "dies randomly sometimes"
    }
}

data.update(new_model_descriptions)

# Write the updated JSON back to the file
with open('existing_file.json', 'w') as f:
    json.dump(data, f, indent=4)
