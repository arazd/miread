from datasets import Dataset, load_dataset, load_metric
import torch
from transformers import AutoTokenizer

import os
import pandas as pd
import numpy as np
#from tqdm.auto import tqdm


class AbstractsDataset_LM_CLS:
    def __init__(self,
                 model_checkpoint, # for tokenizer, e.g. 'distilbert-base-uncased'
                 abst_folder='/scratch/ssd002/home/anastasia/Documents/abstracts/Journal_db_2021/',
                 abstracts = [],
                 max_len = 512,
                 summ_len = 35,
                 ):
        self.abst_folder = abst_folder
        if abstracts == []: 
            self.abstracts = [x for x in sorted(os.listdir(self.abst_folder)) if 'lock' not in x] 
        else:
            self.abstracts = abstracts
        self.num_classes = len(self.abstracts)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.max_len = max_len
        self.summ_len = summ_len
        
    def get_texts_labels(self, 
                         min_ab_per_journal = 100,
                         use_ab_per_journal = 100,
                         use_titles = True):
        texts, titles, labels = [], [], []

        #good_abstracts = [x for x in abstracts if 'lock' not in x] # lock_... .csv is not an abstract folder
        good_abstracts = self.abstracts
        #m = len(abstracts)
        #m = 100

        label = 0
        journal_to_label = {}
        for j in range(self.num_classes):
            if j % 50 == 0:
                print(j)

            df = pd.read_csv(self.abst_folder+good_abstracts[j], 
                             sep='\t', header=None, quoting=csv.QUOTE_NONE)

            #print(df.shape)
            if df.shape[0] > min_ab_per_journal:
                for i, row in df.iloc[:use_ab_per_journal].iterrows():
                    if row[4]==row[4] and row[4]!=None and len(row[4]) > 100: 
                        # if abstract is not None / NaN and contains > 100 characters
                        texts.append(row[4])
                        if use_titles: titles.append(row[1])
                        labels.append(int(label))
                journal_to_label[row[3]] = label
                label += 1

        #len(texts), len(labels)
        self.num_classes = len(set(labels)) 
        self.journal_to_label = journal_to_label
        if not use_titles: titles = None
        return texts, titles, labels
    
    
    def train_test_split(self, data_labels_tuple, train_split=0.8):
        idx = np.arange(len(data_labels_tuple[0]))
        np.random.shuffle(idx)# seed=seed)

        lim = int(len(idx)*train_split) # number of texts / labels that go into training data
        
        train_tuple, test_tuple = [], []
        for x in data_labels_tuple:
            x_train, x_test = list(np.array(x)[idx])[:lim], list(np.array(x)[idx])[lim:]
            train_tuple.append(x_train)
            test_tuple.append(x_test)
            
        return train_tuple, test_tuple
    
    
    def create_huggingface_dataset(self, texts, labels, titles=None):
        N = len(texts)
        
        source = self.tokenizer.batch_encode_plus(texts, 
                                                  max_length= self.max_len, 
                                                  #truncation=True,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')
        input_ids = source['input_ids'].squeeze()
        attention_mask = source['attention_mask'].squeeze()
        
        my_dict = {'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'label': labels,
                   'idx': [i for i in range(N)]}
        
        if titles!=None:
            source = self.tokenizer.batch_encode_plus(['summarize: '+x for x in texts], 
                                                  max_length= self.max_len, 
                                                  #truncation=True,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')
            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()

            my_dict['source_ids'] = source_ids
            my_dict['source_mask'] = source_mask

            target = self.tokenizer.batch_encode_plus(titles, 
                                                      max_length= self.summ_len, 
                                                      #truncation=True,
                                                      pad_to_max_length=True,
                                                      return_tensors='pt')
            target_ids = target['input_ids'].squeeze()
            target_mask = target['attention_mask'].squeeze()
        
            my_dict['target_ids'] = target_ids
            my_dict['target_mask'] = target_mask
            
        dataset = Dataset.from_dict(my_dict)
        
        return dataset
    
    
    def dataset_to_torch(self, dataset, tokenizer, batch_size=8):
        

        #dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dataset.set_format(type='torch',  
                           columns=['input_ids', 'attention_mask',
                                    'source_ids', 'source_mask', 
                                    'target_ids', 'target_mask', 
                                    'label'], 
                           #columns=['input_ids', 'attention_mask', 'label'], 
                           device='cuda')
        dataset = dataset.rename_column("label", "labels")
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
        return dataloader
        
        
    def get_train_test_ds(self, 
                          min_ab_per_journal = 100,
                          use_ab_per_journal = 100,
                          to_torch = True,
                          batch_size = 8,
                          tokenizer = None):
        
        texts, titles, labels = self.get_texts_labels(min_ab_per_journal = min_ab_per_journal,
                                                      use_ab_per_journal = use_ab_per_journal)
        train_tuple, test_tuple = self.train_test_split((texts, titles, labels))
        train_texts, train_titles, train_labels = train_tuple
        test_texts, test_titles, test_labels = test_tuple
        train_ds = self.create_huggingface_dataset(train_texts, train_labels, titles=train_titles)
        test_ds = self.create_huggingface_dataset(test_texts, test_labels, titles=test_titles)
        
        if to_torch:
            if tokenizer == None: tokenizer = self.tokenizer
            train_ds = self.dataset_to_torch(train_ds, tokenizer, batch_size=batch_size)
            test_ds = self.dataset_to_torch(test_ds, tokenizer, batch_size=batch_size)
        return train_ds, test_ds
    
    
    



class AbstractsDataset:
    def __init__(self,
                 abst_folder='/scratch/ssd002/home/anastasia/Documents/abstracts/Journal_db_2021/',
                 abstracts = [],
                 model_checkpoint = 'distilbert-base-uncased' # for tokenizer
                 ):
        self.abst_folder = abst_folder
        if abstracts == []: 
            self.abstracts = [x for x in sorted(os.listdir(self.abst_folder)) if 'lock' not in x] 
        else:
            self.abstracts = abstracts
        self.num_classes = len(self.abstracts)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        
    def get_texts_labels(self, 
                         min_ab_per_journal = 100,
                         use_ab_per_journal = 100):
        texts, labels = [], []

        #good_abstracts = [x for x in abstracts if 'lock' not in x] # lock_... .csv is not an abstract folder
        good_abstracts = self.abstracts
        #m = len(abstracts)
        #m = 100

        label = 0
        journal_to_label = {}
        for j in range(self.num_classes):
            if j % 50 == 0:
                print(j)

            df = pd.read_csv(self.abst_folder+good_abstracts[j], 
                             sep='\t', header=None)

            #print(df.shape)
            if df.shape[0] > min_ab_per_journal:
                for i, row in df.iloc[:use_ab_per_journal].iterrows():
                    texts.append(row[4])
                    labels.append(int(label))
                journal_to_label[row[3]] = label
                label += 1

        #len(texts), len(labels)
        self.num_classes = len(set(labels)) 
        self.journal_to_label = journal_to_label
        return texts, labels
    
    
    def train_test_split(self, texts, labels, train_split=0.8):
        idx = np.arange(len(texts))
        np.random.shuffle(idx)

        lim = int(len(idx)*train_split) # number of texts / labels that go into training data
        train_texts, train_labels = list(np.array(texts)[idx])[:lim], list(np.array(labels)[idx])[:lim]
        test_texts, test_labels = list(np.array(texts)[idx])[lim:], list(np.array(labels)[idx])[lim:]
        
        return train_texts, train_labels, test_texts, test_labels
    
    
    def create_huggingface_dataset(self, texts, labels):
        N = len(texts)
        my_dict = {'sentence': [x for x in texts],
                   'label': labels,
                   'idx': [i for i in range(N)]}

        dataset = Dataset.from_dict(my_dict)
        return dataset
    
    
    def dataset_to_torch(self, dataset, tokenizer, batch_size=8):
        dataset = dataset.map(lambda e: tokenizer(e['sentence'], 
                                                  truncation=True, 
                                                  padding='max_length'), 
                              batched=True)

        #dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], 
                           device='cuda')
        dataset = dataset.rename_column("label", "labels")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return dataloader
        
    def get_train_test_ds(self, 
                          min_ab_per_journal = 100,
                          use_ab_per_journal = 100,
                          to_torch = True,
                          batch_size = 8,
                          tokenizer = None):
        
        texts, labels = self.get_texts_labels(min_ab_per_journal = min_ab_per_journal,
                                              use_ab_per_journal = use_ab_per_journal)
        train_texts, train_labels, test_texts, test_labels = self.train_test_split(texts, labels)
        train_ds = self.create_huggingface_dataset(train_texts, train_labels)
        test_ds = self.create_huggingface_dataset(test_texts, test_labels)
        
        if to_torch:
            if tokenizer == None: tokenizer = self.tokenizer
            train_ds = self.dataset_to_torch(train_ds, tokenizer, batch_size=batch_size)
            test_ds = self.dataset_to_torch(test_ds, tokenizer, batch_size=batch_size)
        return train_ds, test_ds
    
