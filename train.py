import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BartTokenizer, BartForSequenceClassification
from transformers import BartModel, BartForConditionalGeneration, BartPretrainedModel, BartConfig
import transformers, datasets
from accelerate import Accelerator

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric

import model_utils, abstr_dataset
import logging, argparse, os, time, json
from datetime import datetime

seed = 1234

def create_model_tokenizer(num_labels,
                           mpath="sshleifer/distilbart-xsum-12-3",
                           use_encoder_features=False # whether to use encoder-only features
                          ):

    model = BartForConditionalGeneration.from_pretrained(mpath)
    tokenizer = BartTokenizer.from_pretrained(mpath)

    # 2-headed model with LM head and classification head
    modelCLS = model_utils.BartForSequenceClassification2.from_pretrained(mpath,
                                                                      num_labels = num_labels,
                                                                      LM_model = model,
                                                                      use_encoder_features = use_encoder_features)

    return modelCLS, tokenizer


def create_dataset_from_pandas(df,
                               tokenizer,
                               batch_size,
                               num_workers=4,
                               MAX_LEN=512,
                               SUMMARY_LEN=40,
                               shuffle=True):

    dataset = abstr_dataset.AbstractsDataset(df, tokenizer, source_len=MAX_LEN, summ_len=SUMMARY_LEN)
    sampler = abstr_dataset.StatefulSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers)
    return dataloader


def create_or_restore_training_state(dataset_df_path,
                                     batch_size=2,
                                     mpath="sshleifer/distilbart-xsum-12-3",
                                     checkpoint_path='checkpoint.pt',
                                     use_encoder_features=False,
                                     lr=5e-5,
                                     weight_decay=0.0):

    df = pd.read_csv(dataset_df_path) #('journal_db_2021_Q1.csv')
    df_val = pd.read_csv(dataset_df_path.replace('train', 'val'))
    df_lm = pd.read_csv('abstract_experiment_df/df_LM.csv')

    logging.info("Loaded dfs {} and {}".format(dataset_df_path, dataset_df_path.replace('train', 'val')))
    # initial configuration of the model
    num_labels = len(list(df['label'].unique()))
    modelCLS, tokenizer = create_model_tokenizer(num_labels=num_labels,
                                                 mpath=mpath,
                                                 use_encoder_features=use_encoder_features
                                                 # whether to use encoder-only features
                                                )
    # creating dataset from df
    dataloader = create_dataset_from_pandas(df, tokenizer, batch_size)
    dataloader_val = create_dataset_from_pandas(df_val, tokenizer, 2, shuffle=False)
    dataloader_lm = create_dataset_from_pandas(df_lm, tokenizer, 2, shuffle=False)

    optimizer = torch.optim.AdamW(params=modelCLS.parameters(), lr=lr, weight_decay=weight_decay)

    epoch = 0
    cur_loss = 0
    # restore training state if checkpoint exists
    if os.path.exists(checkpoint_path):
        training_state = torch.load(checkpoint_path)

        modelCLS.load_state_dict(training_state['model'])
        optimizer.load_state_dict(training_state['optimizer'])
        dataloader.sampler.load_state_dict(training_state['sampler'])
        epoch = training_state['epoch']
        cur_loss = training_state['cur_loss']
        rng = training_state['rng']
        torch.random.set_rng_state(rng)
        logging.info("training state restored at epoch {} and iteration {}".
              format(epoch, dataloader.sampler.data_counter))
        #print("restored! Epoch and iter are:", epoch, dataloader.sampler.data_counter)

    else:
        logging.info("No checkpoint detected, starting from initial state")
        #print("no ckpt detected")

    return modelCLS, tokenizer, optimizer, dataloader, dataloader_val, dataloader_lm, epoch, cur_loss


# we need to be careful when saving checkpoints since preemption can also
# occur during checkpointing. Therefore, we need to make sure the checkpoint
# file is either kept untouched or successfully updated during this process.
def commit_state(model, optimizer, sampler, dataloader_iter, rng, epoch, cur_loss, checkpoint_path):

    #temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")
    temp_path = checkpoint_path
    training_state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'sampler' : sampler.state_dict(dataloader_iter),
        'epoch': epoch,
        'cur_loss' : cur_loss,
        'rng' : rng
    }

    # first save to temp file
    torch.save(training_state, temp_path)
    # according to the GNU spec of rename, the state of checkpoint_path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    os.replace(temp_path, checkpoint_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path + \
    "epoch =" + str(epoch) + " iter =" + str(sampler.data_counter)
    logging.info(msg)
    #print(msg)

def optimizer_to_dev(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def extract_representations(modelCLS, dataloader_lm, device='cuda'):
    modelCLS.eval()
    repr_dict = {}

    for batch in dataloader_lm:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # encoder-decoder features
            hs = modelCLS.model(batch['input_ids'], batch['attention_mask'])['last_hidden_state'][:,-1,:]

        labels = batch['label'].cpu().detach().numpy()
        for i, l in enumerate(labels):
            feature = hs[i].cpu().detach().numpy().reshape(1, -1)
            if l in repr_dict:
                repr_dict[l] = np.concatenate((repr_dict[l], feature))
            else:
                repr_dict[l] = feature

    return repr_dict


def save_metrics(metric_dict, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for metric in metric_dict:
        #loss_path = os.path.join(save_dir, 'loss.npy')
        logging.info('saving {}'.format(metric))
        metric_path = os.path.join(save_dir, metric)
        if os.path.exists(metric_path):
            arr = np.load(metric_path)
        else:
            arr = np.array([])

        value = metric_dict[metric]
        new_arr = np.concatenate([arr, np.array([value])])
        np.save(metric_path, new_arr)


    # if iteration%5000 == 0:
    #     weight_path = os.path.join(save_dir, 'weights_'+str(iteration)+'.pt')
    #     torch.save(modelCLS.state_dict(), weight_path)


def get_validation_metrics(dataloader_val, dataloader_lm, modelCLS, tokenizer):
    modelCLS.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataloader_iter = iter(dataloader_val)
    nlls_val = []
    metric_acc_val= load_metric("accuracy")
    metric_rouge_val = load_metric("rouge")

    with torch.no_grad():
        for _, batch in enumerate(dataloader_iter):
            # LM
            y = batch['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch['source_ids'].to(device, dtype = torch.long)
            mask = batch['source_mask'].to(device, dtype = torch.long)

            outputs = modelCLS.LM_model(input_ids = ids,
                                        attention_mask = mask,
                                        decoder_input_ids = y_ids,
                                        labels = lm_labels)
            loss_LM = outputs[0]
            nlls_val.append(loss_LM)

            generated_ids = modelCLS.LM_model.generate(
                input_ids = ids,
                attention_mask = mask,
                #min_length=10,
                max_length=40,
                num_beams=4,
                repetition_penalty=3.5,
                length_penalty=2.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            refs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in y] # y contrains tokenized target (title)
            metric_rouge_val.add_batch(predictions=preds, references=refs)

            # CLASSIFICATION
            outputs = modelCLS(input_ids=batch['input_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device),
                               labels=batch['label'].to(device)
                              )

            predictions = torch.argmax(outputs.logits, dim=-1)
            metric_acc_val.add_batch(predictions=predictions, references=batch["label"])


        ppl = torch.exp(torch.stack(nlls_val).sum() / len(nlls_val)) # perplexity
        mrouge = metric_rouge_val.compute() # metric will reset after calling compute
        metric_val_dict = {
            'val_acc.npy': metric_acc_val.compute()['accuracy'],
            'val_rouge1.npy': mrouge['rouge1'].mid[2],
            'val_rouge2.npy': mrouge['rouge2'].mid[2], # 1 - index of rouge-recall, 2 - index of rouge-f1
            'val_rouge_L.npy': mrouge['rougeL'].mid[2], # 1 - index of rouge-recall
            'val_perplexity.npy': ppl.detach().cpu().numpy().flatten()[0],
            'val_loss.npy': torch.mean(torch.tensor(nlls_val)).detach().cpu().numpy().flatten()[0],
        }

        #metric_rouge_lm = load_metric("rouge")
        nlls_lm = []
        for _, batch in enumerate(iter(dataloader_lm)):
            y = batch['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch['source_ids'].to(device, dtype = torch.long)
            mask = batch['source_mask'].to(device, dtype = torch.long)

            outputs = modelCLS.LM_model(input_ids = ids,
                                        attention_mask = mask,
                                        decoder_input_ids = y_ids,
                                        labels = lm_labels)
            loss_LM = outputs[0]
            nlls_lm.append(loss_LM)

            generated_ids = modelCLS.LM_model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=40,
                num_beams=3,
                repetition_penalty=3.5,
                length_penalty=2.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            refs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in y] # y contrains tokenized target (title)
            metric_rouge_val.add_batch(predictions=preds, references=refs)

        ppl = torch.exp(torch.stack(nlls_lm).sum() / len(nlls_lm)) # perplexity
        mrouge = metric_rouge_val.compute()
        metric_val_dict['lm_rouge1.npy'] = mrouge['rouge1'].mid[2]
        metric_val_dict['lm_rouge2.npy'] = mrouge['rouge2'].mid[2] # 1 - index of rouge-recall
        metric_val_dict['lm_rouge_L.npy'] = mrouge['rougeL'].mid[2] # 1 - index of rouge-recall
        metric_val_dict['lm_perplexity.npy'] = ppl.detach().cpu().numpy().flatten()[0]
        metric_val_dict['lm_loss.npy'] = torch.mean(torch.tensor(nlls_lm)).detach().cpu().numpy().flatten()[0]

    return metric_val_dict



def train(modelCLS, tokenizer, optimizer, dataloader, dataloader_val, dataloader_lm, epoch, epoch_loss,
          checkpoint_path, checkpoint_interval, max_epoch, coef_lm, coef_cls, metrics_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    #criterion = nn.CrossEntropyLoss()
    epoch_loss = 0
    start_time = time.time()
    modelCLS = modelCLS.to(device)
    optimizer_to_dev(optimizer, device)

    num_training_steps = (max_epoch-epoch) * len(dataloader)
    start = dataloader.sampler.data_counter // dataloader.batch_size # initial value of progress bar counter
    progress_bar = tqdm(range(num_training_steps), initial=start)

    nlls = []
    metric = load_metric("accuracy")
    metric_rouge = load_metric("rouge")

    while epoch < max_epoch:
        #print('epoch ', epoch)
        modelCLS.train()
        logging.info('Starting epoch '+str(epoch) + '/' + str(max_epoch))
        # nlls = []
        # metric = load_metric("accuracy")
        # metric_rouge = load_metric("rouge")

        rng = torch.random.get_rng_state()
        dataloader_iter = iter(dataloader)
        for _, batch in enumerate(dataloader_iter):

            # LM
            y = batch['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch['source_ids'].to(device, dtype = torch.long)
            mask = batch['source_mask'].to(device, dtype = torch.long)

            outputs = modelCLS.LM_model(input_ids = ids,
                                        attention_mask = mask,
                                        decoder_input_ids = y_ids,
                                        labels = lm_labels)
            loss_LM = outputs[0]
            nlls.append(loss_LM) # append negative log likelihood for subsequent perplexity computation
            #inter_loss += loss_LM.item()

            generated_ids = modelCLS.LM_model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=40,
                num_beams=3,
                repetition_penalty=3.5,
                length_penalty=2.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            refs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in y] # y contrains tokenized target (title)
            metric_rouge.add_batch(predictions=preds, references=refs)
            #mr = metric_rouge.compute(predictions=preds, references=refs)
            #rouge += [mr['rouge2'][i][1] for i in range(len(preds))]

            # CLASSIFICATION
            outputs = modelCLS(input_ids=batch['input_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device),
                               labels=batch['label'].to(device)
                              )

            # TOTAL LOSS
            loss = coef_cls*outputs.loss + coef_lm*loss_LM

            loss.backward()
            #accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["label"])
            progress_bar.update(1)

            cur_time = time.time()
            if cur_time - start_time > checkpoint_interval:
                commit_state(modelCLS, optimizer, dataloader.sampler,
                             dataloader_iter, rng, epoch, epoch_loss, checkpoint_path)
                start_time = cur_time

            # Saving metrics / weights
            if _%250 == 0:
                logging.info("Training Loss" + str(loss_LM.item()))
                #logging.info("Saving " + str(inter_loss) + ' ' + str(metric.compute()))
                #inter_loss = 0

                # save metrics [after each epoch?]
                ppl = torch.exp(torch.stack(nlls).sum() / len(nlls)) # perplexity
                mrouge = metric_rouge.compute()
                metric_dict_train = {
                    'train_acc.npy': metric.compute()['accuracy'],
                    'train_rouge1.npy': mrouge['rouge1'].mid[2],
                    'train_rouge2.npy': mrouge['rouge2'].mid[2], # 1 - index of rouge-recall
                    'train_rouge_L.npy': mrouge['rougeL'].mid[2], # 1 - index of rouge-recall
                    'train_perplexity.npy': ppl.detach().cpu().numpy().flatten()[0],
                    'train_loss.npy': torch.mean(torch.tensor(nlls)).detach().cpu().numpy().flatten()[0],
                }
                metric_dict_val = get_validation_metrics(dataloader_val, dataloader_lm, modelCLS, tokenizer)
                metric_dict = {**metric_dict_train, **metric_dict_val}
                #metric_dict = metric_dict_train
                save_metrics(metric_dict, metrics_dir)
                repr_dict = extract_representations(modelCLS, dataloader_lm, device=device)
                np.save(os.path.join(metrics_dir, 'repr_dict_'+str(epoch)), repr_dict)

                nlls = []
                metric = load_metric("accuracy")
                metric_rouge = load_metric("rouge")
                modelCLS.train()

        msg = "Epoch: {}, Loss: {}".format(epoch, epoch_loss)
        logging.info(msg)

        # save weights at the end of each epoch
        #commit_state(modelCLS, optimizer, dataloader.sampler,
        #             dataloader_iter, rng, epoch, epoch_loss, checkpoint_path)

        epoch += 1
        epoch_loss = 0

    return modelCLS


def main(args):
    args_dict_path = os.path.join(args.metrics_dir, 'commandline_args.txt')
    if not os.path.exists(args.metrics_dir): os.mkdir(args.metrics_dir)
    with open(args_dict_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

    logging.info("starting training script")
    torch.random.manual_seed(seed)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")

    # check if the checkpoint exists and try to resume from the last checkpoint
    # if you are saving for every epoch, you can skip the part about
    # saving and loading the dataloader state.
    modelCLS, tokenizer, optimizer, dataloader, dataloader_val, dataloader_lm, epoch, cur_loss= \
        create_or_restore_training_state(dataset_df_path=args.dataset_df_path,
                                         batch_size=args.batch_size,
                                         mpath=args.mpath,
                                         use_encoder_features=(args.use_encoder_features==1),
                                         checkpoint_path=checkpoint_path,
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

    # now we can start the training loop
    modelCLS = train(modelCLS, tokenizer, optimizer, dataloader, dataloader_val, dataloader_lm, epoch, cur_loss,
                     checkpoint_path, args.checkpoint_interval, args.num_epoch, args.coef_lm, args.coef_cls, args.metrics_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script that performs checkpointing for PyTorch'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        help='number of epochs to run',
        required=True
    )


    parser.add_argument(
        '--dataset_df_path',
        type=str,
        help='path to dataframe with abstracts / titles / labels',
        required=True,
    )

    parser.add_argument(
        '--metrics_dir',
        type=str,
        help='path to folder to save metrics',
        required=True,
    )

    parser.add_argument(
        '--mpath',
        type=str,
        help='NLP model path on HuggingFace (e.g. BART, DistilBART etc.)',
        default="sshleifer/distilbart-xsum-12-3",
    )

    parser.add_argument(
        '--use_encoder_features',
        type=int,
        help='Use full model features (default) vs. encoder-only features',
        default=0,
    )

    parser.add_argument(
        '--coef_lm',
        type=float,
        help='Weighting coefficient for LM loss',
        default=1.0,
    )

    parser.add_argument(
        '--coef_cls',
        type=float,
        help='Weighting coefficient for CLS loss',
        default=1.0,
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=5e-5,
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay (for AdamW)',
        default=0.0,
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='path to save and look for the checkpoint file',
        default=os.path.join(os.getcwd(), "runs")
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size per iteration',
        default=2
    )

    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        help='period to take checkpoints in seconds',
        default=1500 # 25 minutes
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='specify the location of the output directory, default stdout',
        default=None
    )

    main(parser.parse_args())
