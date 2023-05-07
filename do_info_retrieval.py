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


def uncover_pairs(mat, paper_ids):
    N = mat.shape[0]
    flattened_mat = []
    flattened_pairs = []

    progress_bar = tqdm(np.arange(N*N//2) )
    for i in range(N):
        for j in range(i+1,N):
            #print(mat[i,j], genesN[i], genesN[j])
            flattened_mat.append(mat[i,j])
            flattened_pairs.append([paper_ids[i], paper_ids[j]])

            progress_bar.update(1)
    progress_bar.close()

    return flattened_mat, flattened_pairs



def get_y_scores(flattened_pairs, flattened_mat, arxiv_cat_dict):
    y_score = []
    y_true = []

    progress_bar = tqdm(np.arange(len(flattened_pairs)) )
    for k, pair in enumerate(flattened_pairs):
        #pair = flat_pairs[k]
        corr_score = flattened_mat[k]
        y_score.append(corr_score)
        pid1 = pair[0]
        pid2 = pair[1]

        if pid1 in arxiv_cat_dict and pid2 in arxiv_cat_dict:
            list1 = arxiv_cat_dict[pid1]
            list2 = arxiv_cat_dict[pid2]
            if len([x for x in list2 if x in list1])>0:
            #if sorted(list1) == sorted(list2):
                y_true.append(1) # at least 1 category intersects
            else:
                y_true.append(0) # no intersection with labels
        else:
            print("Mistake with ", pid1, pid2)

        progress_bar.update(1)
    progress_bar.close()
    return y_true, y_score



def main(args):
    #assert args.data in ['extra', 'iclr', 'mag', 'arxiv', 'all']
    #assert (args.model_type in ['specter', 'scibert', 'bart', 'pubmedbert', 'citebert', 'biobert', 'bert'] or 'scibert_finetune' in args.model_type)

    df_arxiv = pd.read_csv('abstract_experiment_df/df_arxiv_information_retrieval.csv')
    category_calls = list(df_arxiv['categories'])
    major_categories = ['cs', 'econ', 'physics', 'math', 'eess', 'q-bio', 'q-fin', 'stat']
    categories_dict = {x: [] for x in major_categories}

    for i, c in enumerate(category_calls):
        field = list(set([x.split('.')[0] for x in c]))
        if len(field)==1 and field[0] in categories_dict:
            categories_dict[field[0]].append(i)
    # categories_dict['q-bio'] = [512, 771, ...] -> idx of q-bio papers

    #rep_dict = np.load('/scratch/hdd001/home/anastasia/specter/features_dir/repr_dict_arxiv_specter.npy', allow_pickle=True)[()]
    rep_dict = np.load(args.features_path, allow_pickle=True)[()]
    arr = np.array([rep_dict[i][0] for i in range(len(list(rep_dict)))])
    c = np.corrcoef(arr)
    c = c-np.diag(np.diag(c))

    for cat in categories_dict:
        # cat = 'physics'
        c_field = c[categories_dict[cat]][:, categories_dict[cat]]
        flattened_mat, flattened_pairs = uncover_pairs(c_field, categories_dict[cat])
        y_true, y_score = get_y_scores(flattened_pairs, flattened_mat, arxiv_cat_dict)

        avg_precision = average_precision_score(y_true, y_score)
        avg_auc = roc_auc_score(y_true, y_score)
        res_dict = {
            'avg_auc': avg_auc,
            'avg_precision': avg_precision,
            'y_true': y_true,
            'y_score': y_score,
        }
        #path = os.path.join(args.model_dir, 'validation', 'val_dict_'+args.weights_file.split('.')[0]+'.npy')
        path = args.features_path.replace('/repr_dict', '/y_true_score_'+cat)
        np.save(path, res_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script that performs checkpointing for PyTorch'
    )

    parser.add_argument(
        '--features_path',
        type=str,
        help='path to features dict (e.g. "runs/DistilBART_full_pubmed/")',
        default='/scratch/hdd001/home/anastasia/bart_models/checkpoints_6884277_BART_field_cls2_lm1_lr5e-6/'
    )

    # parser.add_argument(
    #     '--data',
    #     type=str,
    #     help='data to extract representations or all to extract on all datasets (e.g. mag / iclr / extra / all)',
    #     default='arxiv'
    # )


    main(parser.parse_args())
