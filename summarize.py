from pdf_txt import pdf_to_text, get_num_pages, pdf_to_text_miner
import os
import openai
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import json
import requests
import traceback

input_file='sample.pdf'

def openai_summarize(input_file):
    openai.api_key= "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"

    num_count = get_num_pages(input_file)
    f = open('summarized.txt','a+')

    count=0
    while count<num_count:

        text= pdf_to_text(input_file,count)
        #print(text)
        response = openai.Completion.create(
            engine="davinci", 
            prompt= text, 
            temperature=0.3,
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )

        summarized_text = response['choices'][0]['text'] 
        print(summarized_text)
        f.write(summarized_text)
        
    f.close()



def bart_large(input_file):
    #headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

    def query(payload):
        data = json.dumps(payload)
        #, headers=headers
        response = requests.request("POST", API_URL, data=data)
        return json.loads(response.content.decode("utf-8"))
    
    def splitter(n, s):
        pieces = s.split()
        out = (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))
        return out
    #num_count = get_num_pages(input_file)
    f = open('summarized_bart_api.txt','a+')
    #count = 0 

    text = pdf_to_text_miner(input_file)
    #splat = text.split("\n\n")
    #print(splat)
    #print(type(splat))
    #print(len(splat))
    i = 0
    #for paragraph in enumerate(splat,0):
    for paragraph in splitter(1000, text):
        #print(paragraph)
        #print(type(paragraph))
        #print(len(paragraph))
        i+= 1
        #paragraph=str(paragraph)
        #print(paragraph[1])
        #print(type(paragraph[1]))
        #print(len(paragraph[1]))
        
        try:
            print('\n\n')
            print('on paragraph', i)
            print('paragraph length is:',len(paragraph))
            data=query(paragraph)
            #print(data)
            output = data[0]['summary_text']
            #print(output)
            print('summary length for paragraph is:', len(output))
            f.write(output)
            f.write('\n\n')
        except Exception as e:
            error = traceback.format_exc()
            print(error)
            print(data)
        
        """
        if len(paragraph[1])>150:
            try:
                print('\n\n')
                print('on paragraph', paragraph[0])
                print('paragraph length is:',len(paragraph[1]))
                data=query(paragraph[1])
                #print(data)
                output = data[0]['summary_text']
                #print(output)
                print('summary length for paragraph is:', len(output))
                f.write(output)
                f.write('\n\n')
            except Exception as e:
                error = traceback.format_exc()
                print(error)
                print(data)
        else:
            continue
    """
    """    
    data=query(text)
    #print(data)
    output = data[0]['summary_text']
    #print(output)
    f.write(output)
    """
    """
    while count<num_count:
        text= pdf_to_text(input_file,count)
        data = query(text)
        output= data[0]['summary_text']
        print(output)
        f.write(output)
        count += 1
    """
    f.close()

if __name__=='__main__':
    #openai_summarize(input_file)
    #bart_summarize(input_file)
    bart_large(input_file)

