import json
import pandas as pd

# Step 1
with open('generatedText - final.txt', 'r') as file:
    json_string = file.read()
    # json_string = """{ "datasets" : [ """ + json_string + "}"
    json_string = json_string.replace("""```json""", ",").replace("""``````""", ",").replace("""```""", ",").replace(",,", ",").replace("][", "],[")
    json_string = json_string.replace("\s", "").replace("\S", "").replace("\C", "").replace("\A", "").replace("\D", "").replace("\T", "").replace("\I", "").replace("\b", "").replace("\p", "").replace("\w", "").replace("\W", "").replace("\f", "").replace("\c", "").replace("\j", "")
    json_string = json_string.replace("\$", "$").replace("$$", "$")
    # print(json_string)

with open('generatedText - final.txt', 'w') as file:
    file.write(json_string)

#Step 3
with open('generatedText - final.txt', 'r') as file:
    json_string = file.read()

    # Parse the JSON string
    data = json.loads(json_string)
    
    df = pd.DataFrame(columns=["subject", "body"])

    for mail_set in data["datasets"]:
        for i in mail_set:
            df.loc[len(df)] = [i["subject"], i["body"]]
            
df.to_excel("result.xlsx")
            
            