import re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


def remove_stopwords(text_splited_stemmed):
    stop_words = set(stopwords.words('english'))
    filtered_words = {w for w in text_splited_stemmed if w not in stop_words}
    return filtered_words

ps = PorterStemmer()

model = OllamaLLM(model="mistral")

file_path = 'persona.txt'

for i in range(500):
        prompt = ChatPromptTemplate.from_template(
                        """                        
                        Create a persona with a random english name, 2 bad habits, 2 good habits, age, salary, race, job and employer, 4 favorite sports team, 
                        2 favorite store, 4 favorite web sites, 2 hobbies, and 2 favorite country. 

                        Only include english words.

                        Before generating the content, do not repeat the style and the phrases used previously refering to the previous generated words: {sleep_on}
                        Always rephrase the content.

                        Do not generate note.
                        Do not generate + character and " character.

                        The format should look like the following text:
                        John Doe, a 35-year-old Caucasian software engineer at Google with a salary of $120,000, has bad habits of procrastination and biting nails. 
                        He enjoys hiking and painting as hobbies and prefers Argentina and France as his favorite countries. His favorite sports teams are Manchester United 
                        and the New York Yankees, and he loves shopping at Apple Store and Target. John frequently visits ESPN, Reddit, and YouTube.
                        """

                )

        chain = prompt | model

        file = open(file_path, "r+")

        # Extracting the unique words to avoid repeating the phrases
        plain_file = re.sub("\\W", " ", file.read())              
        text_splited = {w.lower() for w in set(plain_file.split())}
        text_splited_nstopwords = remove_stopwords(text_splited)
        text_splited_nstopwords_stemmed =  {ps.stem(w)for w in text_splited_nstopwords}


        response = chain.invoke({
                "sleep_on": str(text_splited_nstopwords)
                })

        # print(len(text_splited_nstopwords))
        # print(len(text_splited_nstopwords_stemmed))
        print(i)

        # print(response)
        file.write("\"\"\""+response+"\"\"\"")
        file.write(',')
        file.close()



