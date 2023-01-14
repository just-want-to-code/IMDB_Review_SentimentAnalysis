from googlesearch import search
import numpy as np
from bs4 import BeautifulSoup
import requests
import torch
import torch.nn as nn
import transformers
import pandas as pd

class config:
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "model.pth"
    # define the tokenizer
    # we use tokenizer and model
    # from huggingface's transformers
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(BERT_PATH,
                                                do_lower_case=True)

class BERTDataset:
    def __init__(self, review):
        """
        :param review: list or numpy array of strings
        :param targets: list or numpy array which is binary
        """
        self.review = review
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    def __len__(self):
        # this returns the length of dataset
        return len(self.review)
    def __getitem__(self, item):
        # for a given item index, return a dictionary
        # of inputs
        review = str(self.review[item])
        review = " ".join(review.split())
        # here, review is a string
        inputs = self.tokenizer.encode_plus(review,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            truncation=True)
        # ids are ids of tokens generated
        # after tokenizing reviews
        ids = inputs["input_ids"]
        # mask is 1 where we have input
        # and 0 where we have padding
        mask = inputs["attention_mask"]
        # token type ids behave the same way as
        # mask in this specific case
        # in case of two sentences, this is 0
        # for first sentence and 1 for second sentence
        token_type_ids = inputs["token_type_ids"]
        # now we return everything
        # note that ids, mask and token_type_ids
        # are all long datatypes and targets is float
        return {
                "ids": torch.tensor(
                ids, dtype=torch.long),
                "mask": torch.tensor(
                mask, dtype=torch.long),
                "token_type_ids": torch.tensor(
                token_type_ids, dtype=torch.long),
                }

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        # we fetch the model from the BERT_PATH 
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,return_dict=False)
        # add a dropout for regularization
        self.bert_drop = nn.Dropout(0.3)
        # a simple linear layer for output
        self.out = nn.Linear(768, 1)
    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # we use the output of the pooler which is of the size
        # (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on
        # if we are using bert base or large respectively
        # in our case, it is 768
        _, o2 = self.bert(ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids)
        # pass through dropout layer
        bo = self.bert_drop(o2)
        # pass through linear layer
        output = self.out(bo)
        # return output
        return output

def eval_fn(data_loader, model, device):
    """
    this is the validation function that generates
    predictions on validation data
    :param data_loader: it is the torch dataloader object
    :param model: torch model, bert in our case
    :param device: can be cpu or cuda
    :return: output and targets
    """
    # put model in eval mode
    model.to(device)
    model.eval()
    # initialize empty lists for
    # targets and outputs
    fin_targets = []
    fin_outputs = []
    # use the no_grad scope
    # its very important else you might
    # run out of gpu memory
    with torch.no_grad():
        # this part is same as training function
        # except for the fact that there is no
        # zero_grad of optimizer and there is no loss
        # calculation or scheduler steps.
        for d in data_loader:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            outputs = model(ids=ids,
                            mask=mask,
                            token_type_ids=token_type_ids)
            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())
    return fin_outputs


class Movie:
  def __init__(self, movie_name):
    self.movie = movie_name

  def get_reviews(self):
    #get reviews as pandas df
    query= "imdb"+self.movie+"user reviews"
    flag=0
    l=[]
    for j in search(query, tld="co.in", num=5, stop=5, pause=2):
      l.append(j)

    url=""
    for i in l:
      if 'reviews' in i:
        url=i 
        break
    
    try:
      r = requests.get(url)
    except:
      flag=1
      return " "," ",flag

    soup = BeautifulSoup(r.content, 'html5lib') 
    ti= soup.find('h3',{'itemprop':'name'})

    title= ti.find('a').get_text()
    date= ti.find('span').get_text()
    date=''.join(date.split())
    title+=date

    table = soup.find('div', attrs = {'class':'lister-list'})
    my_reviews= table.find_all("a", {"class": "title"})

    reviews=[]
    for review in my_reviews:
      reviews.append(review.get_text())

    reviews_df= pd.DataFrame(reviews, columns =['reviews'])

    return title,reviews_df,flag   
      
  def get_sentiment(self,rev,model):
    #get sentiment by passing through model
    sentiment=0
    map={0:'bad', 1:'good'}

    rev_dataset = BERTDataset(review=rev.reviews.values)
    rev_loader = torch.utils.data.DataLoader(rev_dataset,batch_size=config.BATCH_SIZE,num_workers=1)
    device= config.DEVICE 
    outputs= eval_fn(rev_loader, model, device)

    sentiment=np.mean(outputs)
    sentiment= 0 if sentiment<0.5 else 1 
    return map[sentiment]

def get_model():
  model= BERTBaseUncased()
  model.load_state_dict(torch.load('model/model.pth', map_location=config.DEVICE))
  return model

