from ollama import chat
import time
import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("Device Name:", torch.cuda.get_device_name(0))

start = time.time()
 

f = open("generatedText.txt", "w")

for x in range(1000):

    stream = chat(
        model='llama3.2-phisher',
        messages=[{'role': 'user',
                'content': '''I am builing phishing detection model. Please generate me a phishing email to test the model.
                    Pretend that you are legit institution or companies trying to seduce poeple to click the malicious links to expose their personal information or download malware.
                    
                    Topics:
                    1. Urgent Account Verification
                    2. Financial Transaction
                    3. Prize or Lottery Winning
                    4. Fake Invoice or Payment Request
                    5. Charity Scam
                    6. Account Security
                    7. Tax Refund Scam
                    8. Job Offer
                    9. Social Media Notification
                    10. COVID-19 Related Scam
                    11. Law Breaking Activity
                    12. War-Related Aid

                    Generate mail one by one following the topics above and be very convincing to trick the user into clicking on the links or open the malicious files. 
                    Use fake American real names and the existing company name as sender names wihtout parentheses that make the mails look template...
                    The link should be a fake link or file link, do not use real links but looks like a legit link.
                    Do not mention the name of recipient because you do not know the targeted people, use general terms like "Dear Customer" or others diplomatic way to call recipient.
                    The mail content should be concise but lengthier.
                    Diversify the articulation and description style.
                    Make the subject titles more inconsistent.

                    The output should be only in the form of a json file incluing "subject", "body".
                    Focus on generating the email subject and body. Do not generate exaplinations or reminders about the content and the malicious using.
                    Do not generate escape characters.
                    Do not generate note.
                '''}],
        stream=True
    )



    for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            f.write(chunk['message']['content'])

f.close()

done = time.time()
elapsed = done - start

print(elapsed/60, 'mins')