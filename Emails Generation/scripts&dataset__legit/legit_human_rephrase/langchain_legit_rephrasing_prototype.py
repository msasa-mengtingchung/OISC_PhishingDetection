
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import pandas as pd


model = OllamaLLM(model="mistral")

load_path = 'legit_dataset.csv'
output_path = 'legit_rephrasing_dataset.txt'

df = pd.read_csv(load_path)

df = df[3000:4000]


for i, r in df.iterrows():
        
        prompt = ChatPromptTemplate.from_template(
                """

                Rephrase the follow email:
                
                {body}

                Do not generate " sign in the body.
                Keep the response format like the following json format:

                {{ 
                "body": "response"
                }},

                """
        )

        chain = prompt | model


        file = open(output_path, "a")

        response = chain.invoke({
                "body": r["body"]
                })
        
        print(i)
        
        # print(response)
        file.write(response)
        file.write(',')
        file.close()

        
