import json 

with open('yoga_factual_queries.json') as f:
    data = json.load(f)

new_data = {
    "Category": "General",
    "Questions": []
}

for user in data["Users"]:
    for name, info in user.items():
        for question in info["Questions"]:
            new_data["Questions"].append({
                "Question": question,
                "GroundTruth": info["GroundTruth"]
            })
print(len(new_data["Questions"]))

#save the new data to a new json file
with open('new_data.json', 'w') as f:
    json.dump(new_data, f, indent=4)




