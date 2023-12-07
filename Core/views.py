from django.shortcuts import render,redirect
from .forms import PredictForm
from .forms import AddDataForm
from . forms import PredictForm2
# Create your views here.
from playsound import playsound

import random
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

emotions_dict = {
    "id_tag":{
        0: "sadness",
        1: "anger",
        2: "love",
        3: "surprise",
        4: "fear",
        5: "joy"
    }
}

emotions_dict["tag_id"] = dict((j, i) for i, j in emotions_dict["id_tag"].items())
def readData(sub_path):
    path = "Core/data/" + sub_path + ".txt"
    data = open(path).read().split("\n")
    
    x = []
    
    print(f"load {sub_path}...")
    for txt in tqdm(data):
        try:
            text = txt.split(";")
            text[1] = emotions_dict["tag_id"][text[1]]
            x.append(text)
        except:
            continue

    #df = pd.DataFrame(x, columns=["text", "emotion"])
    
    return x

trx = readData("train")
tstx = readData("test")
vx = readData("val")

isAble=True


train=pd.read_csv("Core/data/train.txt",names=['text','emotion'],sep=';')
test=pd.read_csv("Core/data/test.txt",names=['text','emotion'],sep=';')
val=pd.read_csv("Core/data/val.txt",names=['text','emotion'],sep=';')
mytrain=[]

index = train[train.duplicated() == True].index
train.drop(index, axis = 0, inplace = True)
train.reset_index(inplace=True, drop = True)

index = test[test.duplicated() == True].index
test.drop(index, axis = 0, inplace = True)
test.reset_index(inplace=True, drop = True)

index = val[val.duplicated() == True].index
val.drop(index, axis = 0, inplace = True)
val.reset_index(inplace=True, drop = True)

def dataframe_difference(df1, df2, which=None):
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(df):
    df.text=df.text.apply(lambda text : lower_case(text))
    df.text=df.text.apply(lambda text : Removing_numbers(text))
    df.text=df.text.apply(lambda text : Removing_punctuations(text))
    df.text=df.text.apply(lambda text : Removing_urls(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    return sentence

train=normalize_text(train)
test=normalize_text(test)
val=normalize_text(val)

# train model
X_train = train['text'].values
y_train = train['emotion'].values

X_test = test['text'].values
y_test = test['emotion'].values

X_val = val['text'].values
y_val = val['emotion'].values
print(train.emotion.value_counts().get('joy',0))

size =[train.emotion.value_counts().get('joy',0),
       train.emotion.value_counts().get('sadness',0),
       train.emotion.value_counts().get('anger',0),
       train.emotion.value_counts().get('fear',0),
       train.emotion.value_counts().get('love',0),
       train.emotion.value_counts().get('surprise',0)
       ]


def train_model(model,data,targets):
    text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',model)])
    text_clf.fit(data,targets)
    return text_clf


log_reg = train_model(RandomForestClassifier(random_state = 0), X_train, y_train)
pred = log_reg.predict(['Happy'])
print(pred)

def run(request):
    return render(request,'about.html')

def Dataset(request):
    items=random.sample(trx,10)
    if(request.method == "POST"):
        fm=AddDataForm(request.POST)
        if fm.is_valid():
            if(int(fm.cleaned_data['emotion'])==0):
                newData=[fm.cleaned_data['type'],'sadness']
            elif(int(fm.cleaned_data['emotion'])==1):
                newData=[fm.cleaned_data['type'],'anger']
            elif(int(fm.cleaned_data['emotion'])==2):
                newData=[fm.cleaned_data['type'],'love']
            elif(int(fm.cleaned_data['emotion'])==3):
                newData=[fm.cleaned_data['type'],'surprise']
            elif(int(fm.cleaned_data['emotion'])==4):
                newData=[fm.cleaned_data['type'],'fear']
            else:
                newData=[fm.cleaned_data['type'],'joy']

            mytrain.append(newData)
            return redirect('dataset')
    fm=AddDataForm(auto_id=True)
    context={
        "items":items,
        "form":fm,
        "my_items":reversed(mytrain),
        "data_size":len(mytrain)
        }
    return render(request,'dataset.html',context)


def Train(request):
    context={
        "data_size":len(mytrain),
        "size":size
    }
    return render(request,'train.html',context)

def Predict_Views(request):
    pred=""
    if(request.method == 'POST'):
        fm = PredictForm(request.POST)
        if fm.is_valid():
            print('data: ',fm.cleaned_data['query'])
            global log_reg
            predict_arr=log_reg.predict([normalized_sentence(fm.cleaned_data['query'])])
            pred=predict_arr[0]
    else:
        fm = PredictForm(auto_id=True)
    bgm_path = 'media/bgm.mp3'
    img_path='media/welcome.gif'
    if(pred == "sadness"):
        bgm_path='media/sadness.mp3'
        img_path='media/sadness.gif'
    elif(pred == 'fear'):
        bgm_path='media/fear.mp3'
        img_path='media/fear.gif'
    elif(pred == 'joy'):
        bgm_path='media/joy.mp3'
        img_path='media/joy.gif'
    elif(pred == 'love'):
        bgm_path = 'media/love.mp3'
        img_path ='media/love.gif'
    elif(pred == 'anger'):
        bgm_path='media/anger.mp3'
        img_path='media/anger.gif'
    elif(pred == 'surprise'):
        bgm_path='media/surprise.mp3'
        img_path='media/surprise.gif'
    #fm2=PredictForm2()
    global isAble
    context={
        "pred":pred,
        "form":fm,
        "bgm_path":bgm_path,
        "img_path":img_path,
        "isAble":isAble
    }
    return render(request,'predict.html',context)

def Reset(request):
    mytrain.clear()
    return redirect('dataset')

def Pop_data(request):
    mytrain.pop()
    return redirect('dataset')

def train_data(request):
    
    my_train=pd.DataFrame(mytrain, columns=["text", "emotion"])
    my_train=normalize_text(my_train)
    X_train=my_train['text'].values
    y_train=my_train['emotion'].values
    global log_reg
    log_reg = train_model(RandomForestClassifier(random_state = 0), X_train, y_train)
    global isAble
    isAble = False
    return redirect('predict')