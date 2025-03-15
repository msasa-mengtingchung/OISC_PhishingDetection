
import json, re
import pandas as pd

# file_input = 'mistral_stemmedwords.txt'

# file = open(file_input, "r+").read()
# file = file.replace("\\n", " ")
# file = re.sub("\\n", " ", file)
# file = re.sub(r'\[[^\]]+\]', " ", file)
# file = re.sub(r"\"\s+\+\s+\"", " ", file)
# file = re.sub(r"\s+|\t", " ", file)
# file = re.sub(r"\\", " ", file)
# file = "{\"datasets\":[" + file 

file_output = 'mistral_stemmedwords_plain.txt'

file_final = open(file_output, "r")

data = json.load(file_final)

df = pd.DataFrame(columns=['body'])

for i in data["datasets"]:
    df.loc[len(df)] = {'body':i['body']}

df.to_excel("mistral_reflect_on_features.xlsx")