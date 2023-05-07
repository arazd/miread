import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BartTokenizer, AutoTokenizer
from transformers import BartModel, AutoModel, BertForSequenceClassification

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric

import model_utils, abstr_dataset
import logging, os, time, argparse
from datetime import datetime




def extract_representations_iclr(my_model, df_iclr, tokenizer,
                                 device='cuda', source_len=512, model_type='bart'):
    my_model.eval()
    repr_dict = {}

    progress_bar = tqdm(range(df_iclr.shape[0]))
    for i, row in df_iclr.iterrows():
        pid = row['id']
        abstr = row['abstr']
        title = row['title']

        if abstr!=None and abstr==abstr:
            if model_type=='specter' or 'title' in model_type:
                abstr = title + tokenizer.sep_token + abstr
            #elif model_type=='scibert':
            #    abstr = abstr

            inputs = tokenizer(abstr,
                               max_length = source_len,
                               pad_to_max_length=True,
                               truncation=True,
                               return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                # encoder-decoder features
                if model_type == 'bart':
                    hs = my_model.model(inputs['input_ids'], inputs['attention_mask'])['last_hidden_state'][:,-1,:]
                    repr_dict[pid] = hs[0].cpu().detach().numpy().reshape(1, -1)
                else:
                    result = my_model(**inputs).last_hidden_state[:, 0, :]
                    repr_dict[pid] = result[0].cpu().detach().numpy().reshape(1, -1)
            progress_bar.update(1)
    progress_bar.close()
    return repr_dict



def extract_representations_meg_mash(my_model, df, df_meta, tokenizer, model_type,
                                     device='cuda', source_len=512):
    my_model.eval()
    repr_dict = {}

    progress_bar = tqdm(range(df.shape[0]))
    for i, row in df.iterrows():
        pid = row['pid']
        if pid in list(df_meta) and df_meta[pid]['abstract'] != None and df_meta[pid]['abstract']==df_meta[pid]['abstract'] and pid in list(df_meta):
            abstr = df_meta[pid]['abstract']

            if model_type=='specter' or 'title' in model_type:
                title = df_meta[pid]['title']
                abstr = title + tokenizer.sep_token + abstr

            inputs = tokenizer(abstr,
                               max_length = source_len,
                               pad_to_max_length=True,
                               truncation=True,
                               return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if model_type in 'bart':
                with torch.no_grad():
                    # encoder-decoder features
                    hs = my_model.model(inputs['input_ids'], inputs['attention_mask'])['last_hidden_state'][:,-1,:]
                repr_dict[pid] = hs[0].cpu().detach().numpy().reshape(1, -1)

            else: # scibert / scibvert finetune / pubmedbert
                with torch.no_grad():
                    result = my_model(**inputs).last_hidden_state[:, 0, :]
                repr_dict[pid] = result[0].cpu().detach().numpy().reshape(1, -1)
            progress_bar.update(1)
    progress_bar.close()
    return repr_dict


def get_model_tokenzier(args, device='cuda'):
    #assert args.model_type in ['specter', 'scibert', 'bart']
    if args.model_type == 'specter':
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter')
        model.to(device)

    elif args.model_type == 'scibert':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        model.to(device)

    elif args.model_type == 'pubmedbert':
        mpath = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = AutoModel.from_pretrained(mpath)
        model.to(device)

    elif args.model_type == 'biobert':
        mpath = 'dmis-lab/biobert-base-cased-v1.1'
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = AutoModel.from_pretrained(mpath)
        model.to(device)

    elif args.model_type == 'bert':
        mpath = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = AutoModel.from_pretrained(mpath)
        model.to(device)

    elif args.model_type == 'citebert':
        mpath = 'copenlu/citebert'
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = AutoModel.from_pretrained(mpath)
        model.to(device)

    elif args.model_type == 'bart':
        #checkpoint_path = '/scratch/hdd001/home/anastasia/bart_models/checkpoints_6884275_BART_field_cls1_lm0_lr5e-6/weights_0_125005.pt'
        checkpoint_path = os.path.join(args.model_dir, args.weights_file)
        model = torch.load(checkpoint_path, map_location=torch.device(device))
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")

    elif 'scibert_finetune' in args.model_type:
        #checkpoint_path = '/scratch/hdd001/home/anastasia/bart_models/checkpoints_6884275_BART_field_cls1_lm0_lr5e-6/weights_0_125005.pt'
        mpath = 'allenai/scibert_scivocab_uncased'
        model = BertForSequenceClassification.from_pretrained(mpath, num_labels = 2734)
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        checkpoint_path = os.path.join(args.model_dir, args.weights_file)
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.bert
        model.to(device)

    return model, tokenizer



def get_basepath(args):
    save_suffix = '_'+ args.model_type
    if args.model_type == 'scibert':
        base_path = '/scratch/hdd001/home/anastasia/scibert/'

    elif 'scibert_finetune' in args.model_type:
        #base_path = '/scratch/hdd001/home/anastasia/scibert_finetune/'
        base_path = '/scratch/hdd001/home/anastasia/' + args.model_type
        #base_path = args.model_dir
        save_suffix = str(args.weights_file).replace('weights', '').replace('.pt', '')

    elif args.model_type == 'specter':
        base_path = '/scratch/hdd001/home/anastasia/specter/'

    elif args.model_type == 'pubmedbert':
        base_path = '/scratch/hdd001/home/anastasia/pubmedbert/'

    elif args.model_type == 'bert':
        base_path = '/scratch/hdd001/home/anastasia/bert/'

    elif args.model_type == 'biobert':
        base_path = '/scratch/hdd001/home/anastasia/biobert/'

    elif args.model_type == 'citebert':
        base_path = '/scratch/hdd001/home/anastasia/citebert/'

    elif args.model_type == 'bart':
        base_path = args.model_dir
        save_suffix = str(args.weights_file).replace('weights', '').replace('.pt', '')

    base_path = os.path.join(base_path, 'features_dir')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    return base_path, save_suffix



def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    assert args.data in ['extra', 'iclr', 'mag', 'arxiv', 'all']
    assert (args.model_type in ['specter', 'scibert', 'bart', 'pubmedbert', 'citebert', 'biobert', 'bert'] or 'scibert_finetune' in args.model_type)

    #if args.model_type=='bart' and args.weights_file=='all':
    #    weights_files = [x for x in sorted(os.listdit(args.checkpoint))]

    model, tokenizer = get_model_tokenzier(args, device=device)
    base_path, save_suffix = get_basepath(args) # directory to save features


    if args.data == 'extra' or args.data == 'all':
        df_extra_train = pd.read_csv('abstract_repr_df/df_extra_train.csv')
        df_extra_test = pd.read_csv('abstract_repr_df/df_extra_test.csv')
        repr_dict_train = extract_representations_iclr(model, df_extra_train, tokenizer,
                                                       device=device, source_len=512, model_type=args.model_type)
        repr_dict_test = extract_representations_iclr(model, df_extra_test, tokenizer,
                                                      device=device, source_len=512, model_type=args.model_type)
        np.save(os.path.join(base_path, 'repr_dict_train_extra'+save_suffix+'.npy'), repr_dict_train)
        np.save(os.path.join(base_path, 'repr_dict_test_extra'+save_suffix+'.npy'), repr_dict_test)


    # if args.data == 'iclr' or args.data == 'all':
    #     df_iclr = pd.read_csv('data_reviews/df_iclr.csv')
    #     repr_dict = extract_representations_iclr(model, df_iclr, tokenizer,
    #                                              device=device, source_len=512, model_type=args.model_type)
    #     np.save(os.path.join(base_path, 'repr_dict_iclr'+save_suffix+'.npy'), repr_dict)

    if args.data == 'arxiv' or args.data == 'all':
        df_iclr = pd.read_csv('abstract_experiment_df/df_arxiv_information_retrieval.csv')
        repr_dict = extract_representations_iclr(model, df_iclr, tokenizer,
                                                 device=device, source_len=512, model_type=args.model_type)
        np.save(os.path.join(base_path, 'repr_dict_arxiv'+save_suffix+'.npy'), repr_dict)


    if args.data == 'mag' or args.data == 'mesh' or args.data == 'all':
        df_meta = pd.read_json('data/paper_metadata_mag_mesh.json')

        for dataset_name in ['mag', 'mesh']:
            df_train = pd.read_csv('data/'+dataset_name+'/train.csv')
            df_val = pd.read_csv('data/'+dataset_name+'/val.csv')
            repr_dict_val =  extract_representations_meg_mash(model, df_val, df_meta, tokenizer,
                                                              model_type=args.model_type,
                                                              device=device, source_len=512)
            repr_dict_train =  extract_representations_meg_mash(model, df_train, df_meta, tokenizer,
                                                                model_type=args.model_type,
                                                                device=device, source_len=512)
            np.save(os.path.join(base_path, 'repr_dict_train_'+dataset_name+save_suffix+'.npy'), repr_dict_train)
            np.save(os.path.join(base_path, 'repr_dict_test_'+dataset_name+save_suffix+'.npy'), repr_dict_val)


    #path = os.path.join(args.model_dir, 'validation', 'val_dict_'+args.weights_file.split('.')[0]+'.npy')
    #np.save(path, val_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script that performs checkpointing for PyTorch'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        help='model type (e.g. scibert / bart / specter)',
        required=True
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        help='model directory (e.g. "runs/DistilBART_full_pubmed/")',
        default='/scratch/hdd001/home/anastasia/bart_models/checkpoints_6884277_BART_field_cls2_lm1_lr5e-6/'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='data to extract representations or all to extract on all datasets (e.g. mag / iclr / extra / all)',
        default='all'
    )

    parser.add_argument(
        '--weights_file',
        type=str,
        help='model weights file (e.g. "weights_10.pt")',
        default='weights_0_178004.pt'
    )


    main(parser.parse_args())
