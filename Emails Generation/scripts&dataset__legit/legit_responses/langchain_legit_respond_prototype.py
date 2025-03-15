
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import pandas as pd


model = OllamaLLM(model="mistral")

load_path = 'legit_dataset.csv'
output_path = 'legit_responses_dataset.txt'

df = pd.read_csv(load_path)

df_top1000 = df[0:1000]


for i, r in df_top1000.iterrows():
        
        prompt = ChatPromptTemplate.from_template(
                """
                Keep the response format like this in json format:

                {{ 
                "body": "response"
                }},

                Respond the follow email:
                
                {body}

                """
        )

        chain = prompt | model


        file = open(output_path, "a")

        response = chain.invoke({
                "body": r["body"]
                })
        
        print(response)
        
        # print(response)
        file.write(response)
        file.write(',')
        file.close()

        
