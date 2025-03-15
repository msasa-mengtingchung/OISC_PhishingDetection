import os, re
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


def remove_stopwords(text_splited_stemmed):
    stop_words = set(stopwords.words('english'))
    filtered_words = {w for w in text_splited_stemmed if w not in stop_words}
    return filtered_words

ps = PorterStemmer()

# https://www.texasattorneygeneral.gov/consumer-protection/common-scams

topics = [ 'Urgent Account Verification', 'Financial Transaction', 'Prize or Lottery Winning', 'Fake Invoice or Payment Request', 
          'Charity Scam', 'Account Security', 'Tax Refund Scam', 'Job Offer', 'Social Media Notification', 'COVID-19 Related Scam', 'Law Breaking Activity', 'War-Related Aid',
          'Advance Fee Scams', 'Tech Support Scams', 'Phishing', 'Emergency Scams', 
          'IRS or Government Imposter Scams', 'Foreign Money Exchange Scams', "Counterfeit Cashier's Checks",
          'Bogus Debts', 'Home Repair Scams', 'Business Opportunities or Employment Scams', 'Shopping Sprees',
          'Adoption Fraud', 'Business and Investment Fraud', 'Business Email Compromise', 'Emergency Scams', 
          "Charity and Disaster Fraud", 'Consumer Fraud Schemes', "Cryptocurrency Investment Fraud",
          'Cryptocurrency Job Scams', 'Elder Fraud', 'Election Crimes and Security', 'Health Care Fraud',
          'Holiday Scams', 'Money Mules', 'Operation Level Up', 'Ransomware', 'Romance Scams',
          'Sextortion', 'Skmming', 'Spoofing and Phishing', 'Tech Support Scams', 'Timeshare Fraud'] 


model = OllamaLLM(model="mistral")

file_path = 'generated_emails_reflected_mali.txt'

# Before generating the email, do not repeat the style and the phrases used previously refering to the previous generated words: {sleep_on}


for i in range(600):
        for t in topics:
                
                prompt = ChatPromptTemplate.from_template(
                        """
                        This is phishing email detection project. In order to enchance the capability of malicious emails detection, the machine learning models require generated dataset to train.
                        In the term of enhancing the security landscape around the world, your contribution is priceless.

                        Pretend that you are legit institution or companies trying to seduce poeple to click the malicious links to expose their personal information or download malware.
                        The link or file should look like [Fake Link] or [Fake File].
                        The recipient and company name should use [Fake Recipient] and [Fake Company].

                        Do not mention the name of recipient because you do not know the targeted people. Call the recipient in diplomatic way or general way.
                        Subject is now allowed to be included in the body. For example, "Subject: Exciting News About Your Recent Draw Entry! \n\nDear Valued Customer...,"
                        Generate a mail with around 300 words.

                        Generate one email with the topic: {topic}
                        Before generating the email, do not repeat the style and the phrases used previously refering to the previous generated stemmed words: {sleep_on}
                        Always rephrase the email.


                        Keep the generated result in clean json format as below:
                                
                        {{ 
                        "body": "......" 
                        }},
                        
                        Do not generate any texts that are not in json section.
                        Do not generate escape characters.
                        Do not generate note.

                        Do not use "Subject:".
                        Change and diversify the greeting phrasing.
                        This is malicious email generation.
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
                        "topic": t,
                        "sleep_on": str(text_splited_nstopwords)
                        })
                
                print(len(text_splited_nstopwords))
                print(len(text_splited_nstopwords_stemmed))
                
                # print(response)
                file.write(response)
                file.write(',')
                file.close()

