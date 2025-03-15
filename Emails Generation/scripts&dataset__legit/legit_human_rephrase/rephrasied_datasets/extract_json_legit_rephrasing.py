
import json, re
import pandas as pd

file_input = 'legit_rephrased_dataset_combined.txt'

# updated_content = open(file_input, "r", encoding="utf-8").read()
# # updated_content = str(updated_content)
# updated_content = updated_content.replace("\\n", " ")
# updated_content = re.sub("\\n", " ", updated_content)
# updated_content = re.sub(r'\[[^\]]+\]', " ", updated_content)
# updated_content = re.sub(r"\"\s+\+\s+\"", " ", updated_content)
# updated_content = re.sub(r"\s+|\t", " ", updated_content)
# updated_content = re.sub(r"\\", " ", updated_content)
# updated_content = "{\"datasets\":[" + updated_content + "]}"

# with open(file_input, 'w', encoding="utf-8") as file:
#     file.write(updated_content)

file_final = open(file_input, "r", encoding="utf-8")

data = json.load(file_final)

df = pd.DataFrame(columns=['body'])

for i in data["datasets"]:
    df.loc[len(df)] = {'body':i['body']}

df.to_excel("generated_legit_rephrased.xlsx")