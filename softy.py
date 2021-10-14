# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:31:25 2020

@author: RILGRAIN
"""

import socket
socket.getaddrinfo('localhost', 8080)
from tika import parser
import tkinter as tk
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import nltk#this is downloaded
from tkinter import ttk
import tkinter.scrolledtext as scr_txt
#m=tkinter.Tk(screenName=None, baseName=None, className='KeyExtractor', useTk=1)

stop_words = set(stopwords.words("english"))#the stopwords list
punct = set(string.punctuation)#the punctuation list




m = tk.Tk()
m.geometry('1200x700')
m.title("KeyExtractor")


lbl = tk.Label(m, text = "Enter Text Here: ", font=("Arial Bold", 50))
lbl.grid(column=0, row=0)

lbl0 = tk.Label(m, text = "Keyphrase List: ", font=("Arial Bold", 20))
lbl0.grid(column=5, row=0)

e1 = tk.Text(m)
e1 = scr_txt.ScrolledText(m, undo =True)

e1.grid(column=0, row=1)


def uploaddoc():
    filename = tk.filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("text file","*.txt"),("pdf file","*.pdf"),("docx file","*.docx"), ("doc file","*.doc"),("all files","*.*")))
    
    raw = parser.from_file(filename)
    
    
    filename2 = raw['content']
    
    filename = str(filename2)
    safetext = filename.encode('utf-8', errors='ignore')
    safetext = str(safetext).replace('\\n', '')
    for x in safetext:
        e1.insert("end",x)
    
    
btn = tk.Button(m, text= "Upload document", command= uploaddoc)
btn.grid(column=0, row=3)

e1 = tk.Text(m)
e1.grid(column=0, row=1)

e2 = tk.Listbox(m)
e2.grid(column=5, row=1)


def retrieveTxt():
    global top
    text = e1.get("1.0", 'end-1c')
    text = text.lower()
    text1 = re.sub(r'\d+', '', text)#removing numbers
    text2 = nltk.word_tokenize(text1)#tokenized words
    s_text = ["/" if word in stop_words else word for word in text2]#removing the stopwords
    p_text = ["/" if word in punct else word for word in s_text]#removing the punctuation
    
    doc = (p_text)
    
    newdoc = ' ' .join(map(str, doc))#changing it to strings
    new = newdoc
    newd = '/'.join(re.sub(r'(\w)(\s{1})(\w)',r'\1_\3', new).split()).replace('_',' ')#keeping the space between phrases
    newt= '/'.join([i for i in newd.replace("", "").split('/')if i])#total spliting to words and phrases
    
    clean_text = newt.split('/')#turning to strings

    #noun phrase extraction starts here
    noun_text = list(set(clean_text))

    noun_tag = nltk.pos_tag(noun_text)
    #pos_tags = ('NN', 'NNP', 'NNS', 'NNPS')
    nouns = [word for (word,pos) in noun_tag if pos in ['NN', 'NNP', 'NNS', 'NNPS']]#noun_phrases
    noun_phrase = nouns
    #individual nouns
    d = set(doc)
    d_tag = nltk.pos_tag(d)
    dnew = [word for (word,pos) in d_tag if pos in ['NN', 'NNP', 'NNS', 'NNPS']] #nouns in the text
    #comparison of nouns
    noun_phrase = list(map(str.split, noun_phrase))# individual noun phrases
    nscores = [len(set(dnew).intersection(k))/len(k) for k in noun_phrase]#noun phrase scores
    nn = dict(zip(nouns,nscores))# The noun phrases and scores
    #textrank
    text_len = len(noun_text)

    weighted_edge = np.zeros((text_len,text_len),dtype=np.float32)

    score = np.zeros((text_len),dtype=np.float32)
    window_size = 3
    covered_coocurrences = []
    
    for i in range(0,text_len):
        score[i]=1
        for j in range(0,text_len):
            if j==i:
                weighted_edge[i][j]=0
            else:
                for window_start in range(0,(len(clean_text)-window_size+1)):
                    
                    window_end = window_start+window_size
                
                    window = clean_text[window_start:window_end]
                
                    if (noun_text[i] in window) and (noun_text[j] in window):
                    
                        index_of_i = window_start + window.index(noun_text[i])
                        index_of_j = window_start + window.index(noun_text[j])
                    
                        # index_of_x is the absolute position of the xth term in the window 
                        # (counting from 0) 
                        # in the processed_text
                      
                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                            covered_coocurrences.append([index_of_i,index_of_j])


    inout = np.zeros((text_len),dtype=np.float32)
    for i in range(0,text_len):
        for j in range(0,text_len):
            inout[i]+=weighted_edge[i][j]

    MAX_ITERATIONS = 50
    d=0.85
    threshold = 0.0001 #convergence threshold

    for iter in range(0,MAX_ITERATIONS):
        prev_score = np.copy(score)
    
        for i in range(0,text_len):
        
            summation = 0
            for j in range(0,text_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j]/inout[j])*score[j]
                
            score[i] = (1-d) + d*(summation)
    
        if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
            print ("Converging at iteration "+str(iter)+"....")
            break
    for i in range(0,text_len):
        print ("Score of "+noun_text[i]+": "+str(score[i]))


    tt = dict(zip(noun_text,score))

    #nn dict1
    #tt dict2
  
    # adding the values with common key
    for key in tt: 
        if key in nn: 
            tt[key] = tt[key] + nn[key] 
        else: 
            pass
          
    ntt = {k: v for k, v in sorted(tt.items(), key=lambda item: item[1], reverse = True)}
    phraselist = list()
    for key in ntt.keys():
        phraselist.append(key)
    top = phraselist[:10]
    for x in top:
        e2.insert("end", x + '\n')
    print (top)
    print(phraselist)
    print(ntt)
    return top

btn0 = tk.Button(m, text="Extract", command= retrieveTxt, bg= "green")
btn0.grid(column=0, row=4)


def clear_s():
    e1.delete('1.0', 'end')
    e2.delete(0,'end')

btn1 = tk.Button(m, text="Stop/Clear", command= clear_s, bg= "yellow")
btn1.grid(column=0, row=5)

btn2 = tk.Button(m, text='Exit', bg= "Red", command=m.destroy)
btn2.grid(column=0, row=6)





m.mainloop()