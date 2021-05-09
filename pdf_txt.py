import PyPDF2
import textract
#import pdfminer
from pdfminer.converter import TextConverter#
from pdfminer.layout import LAParams#
#from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#
from pdfminer.pdfpage import PDFPage#
#from pdfminer.pdfparser import PDFParser
import io

input_file='sample.pdf'



def get_num_pages(input_file):
    pdf_file = open(input_file, 'rb')
    pypdf_file = PyPDF2.PdfFileReader(pdf_file)
    num_pages = pypdf_file.numPages
    return num_pages


def pdf_to_text_miner(input_file):
    inFile=open(input_file,'rb')
    resMgr= PDFResourceManager()
    retData = io.StringIO()
    TxtConverter = TextConverter(resMgr, retData, laparams=LAParams())
    interpreter=PDFPageInterpreter(resMgr, TxtConverter)

    for page in PDFPage.get_pages(inFile):
        interpreter.process_page(page)
    
    txt= retData.getvalue()

    #print(txt)
    return txt
    #f = open('extracted_miner.txt','w+')
    #f.write(txt)
    #f.close()
    #with open('output_miner.txt','w+') as f:
    #    f.write(txt)

 

def pdf_to_text(input_file,i):
    pdf_file = open(input_file, 'rb')

    #a readable object for PyPDF
    pypdf_file = PyPDF2.PdfFileReader(pdf_file)

    num_pages = i
    count= 0 
    text=''

    """
    while count<num_pages:
        page=pypdf_file.getPage(count)
        count+=1
        text += page.extractText()
    """
    page=pypdf_file.getPage(i)
    text = page.extractText()

    #if PyPDF worked, return text
    if text != '':
    #if text == 2:
        return text
    #PyPDF didnt work, must mean its a scanned page, use textract instead
    else:
        text = textract.process(input_file, method='tesseract', language='eng')
        return text

if __name__=='__main__':
    """
    extracted_text = pdf_to_text(input_file)
    #extracted_text = pdf_miner(input_file)
    print(extracted_text)
    f = open('extracted_text.txt','w+')
    f.write(extracted_text)
    f.close()
    """
    pdf_to_text_miner(input_file)