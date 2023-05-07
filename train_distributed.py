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
import logging, argparse, os, time
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


def create_or_restore_training_state(batch_size=2,
                                     mpath="sshleifer/distilbart-xsum-12-3",
                                     checkpoint_path='checkpoint.pt',
                                     use_encoder_features=False,
                                     lr=5e-5):

    if os.path.exists(checkpoint_path):
        training_state = torch.load(checkpoint_path)
        epoch = training_state['epoch']
    else:
        epoch = 0
    df = pd.read_csv('./abstract_df/df_train_'+str(epoch)+'.csv') # read df corresponding to the current epoch
    logging.info("loading df_train_{}".format(epoch))
    # initial configuration of the model

    num_labels = 4015 #len(list(df['label'].unique()))
    logging.info("num_labels {}".format(num_labels))
    df = df.iloc[:10000]
    modelCLS, tokenizer = create_model_tokenizer(num_labels=num_labels,
                                                 mpath=mpath,
                                                 use_encoder_features=use_encoder_features
                                                 # whether to use encoder-only features
                                                )
    # creating dataset from df
    MAX_LEN = 512
    SUMMARY_LEN = 40
    dataset = abstr_dataset.AbstractsDataset(df, tokenizer, source_len=MAX_LEN, summ_len=SUMMARY_LEN)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=4)
    optimizer = torch.optim.AdamW(params=modelCLS.parameters(), lr=lr)

    epoch = 0
    cur_loss = 0
    # restore training state if checkpoint exists
    if os.path.exists(checkpoint_path):
        training_state = torch.load(checkpoint_path)

        modelCLS.load_state_dict(training_state['model'])
        optimizer.load_state_dict(training_state['optimizer'])
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

    return modelCLS, tokenizer, optimizer, dataloader, epoch, cur_loss


# we need to be careful when saving checkpoints since preemption can also
# occur during checkpointing. Therefore, we need to make sure the checkpoint
# file is either kept untouched or successfully updated during this process.
def commit_state(model, optimizer, rng, epoch, cur_loss, checkpoint_path):

    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")

    training_state = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
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
    "epoch =" + str(epoch) #+ " iter =" + str(sampler.data_counter)
    logging.info(msg)
    #print(msg)

def optimizer_to_dev(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_metrics(loss_val, acc_val, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    loss_path = os.path.join(save_dir, 'loss.npy')
    if os.path.exists(loss_path):
        arr = np.load(loss_path)
    else:
        arr = np.array([])
    new_arr = np.concatenate([arr, np.array([loss_val])])
    np.save(loss_path, new_arr)

    acc_path = os.path.join(save_dir, 'acc.npy')
    if os.path.exists(acc_path):
        arr = np.load(acc_path)
    else:
        arr = np.array([])
    new_arr = np.concatenate([arr, np.array([acc_val])])
    np.save(acc_path, new_arr)



def save_weights(modelCLS, epoch, save_dir):
    weight_path = os.path.join(save_dir, 'weights_'+str(epoch)+'.pt')
    torch.save(modelCLS.state_dict(), weight_path)



def train(modelCLS, tokenizer, optimizer, dataloader, epoch, epoch_loss,
          checkpoint_path, max_epoch, coef_cls, coef_lm):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    #criterion = nn.CrossEntropyLoss()
    epoch_loss = 0

    modelCLS = modelCLS.to(device)
    optimizer_to_dev(optimizer, device)

    num_training_steps = (max_epoch-epoch) * len(dataloader)
    progress_bar = tqdm(range(num_training_steps), initial=0)
    modelCLS.train()

    inter_loss = 0
    while epoch < max_epoch:
        #print('epoch ', epoch)
        logging.info('Starting epoch '+str(epoch) + '/' + str(max_epoch))
        metric = load_metric("accuracy")

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
                                        decoder_input_ids=y_ids,
                                        labels=lm_labels)
            loss_LM = outputs[0]
            inter_loss += loss_LM.item()

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

            # Saving metrics / weights
            if _%50 == 0:
                logging.info("Training Loss" + str(loss_LM.item()))
                if _%500 == 0:

                    #logging.info("Saving " + str(inter_loss) + ' ' + str(metric.compute()))
                    save_dir = checkpoint_path.replace('/checkpoint.pt','_saved')
                    save_metrics(round(inter_loss / 500, 2),
                                 metric.compute()['accuracy'],
                                 save_dir)
                    inter_loss = 0

            epoch_loss += float(loss)

        #print('train accuracy: ', metric.compute())
        msg = "Epoch: {}, Loss: {}, Train accuracy: {}".format(epoch, epoch_loss, metric.compute())
        logging.info(msg)
        epoch += 1
        epoch_loss = 0

        # saving at the end of each epoch
        commit_state(modelCLS, optimizer, rng, epoch, epoch_loss, checkpoint_path)
        save_weights(modelCLS, epoch, save_dir)

        # loading dataframe for the new epoch
        df = pd.read_csv('./abstract_df/df_train_'+str(epoch)+'.csv')
        MAX_LEN = 512
        SUMMARY_LEN = 40
        batch_size = 2
        dataset = abstr_dataset.AbstractsDataset(df, tokenizer, source_len=MAX_LEN, summ_len=SUMMARY_LEN)
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4)

    return modelCLS



# multi-gpu training
def train_with_accelerator(modelCLS, tokenizer, optimizer, dataloader, epoch, epoch_loss,
                           checkpoint_path, max_epoch, coef_cls, coef_lm):
    logging.info('Training with Accelerator')
    accelerator = Accelerator()

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    num_training_steps = (max_epoch-epoch) * len(dataloader)
    progress_bar = tqdm(range(num_training_steps), initial=0, disable=not accelerator.is_main_process)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    modelCLS, optimizer, dataloader = accelerator.prepare(modelCLS, optimizer, dataloader)

    inter_loss = 0
    while epoch < max_epoch:
        #print('epoch ', epoch)
        logging.info('Starting epoch '+str(epoch) + '/' + str(max_epoch))
        metric = load_metric("accuracy")

        rng = torch.random.get_rng_state()
        dataloader_iter = iter(dataloader)
        for _, batch in enumerate(dataloader_iter):

            # LM
            y = batch['target_ids']
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = batch['source_ids']
            mask = batch['source_mask']

            outputs = modelCLS.LM_model(input_ids = ids,
                                        attention_mask = mask,
                                        decoder_input_ids=y_ids,
                                        labels=lm_labels)
            loss_LM = outputs[0]
            inter_loss += loss_LM.item()

            # CLASSIFICATION
            outputs = modelCLS(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               labels=batch['label']
                              )

            # TOTAL LOSS
            loss = coef_cls*outputs.loss + coef_lm*loss_LM

            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["label"])
            progress_bar.update(1)

            # Saving metrics / weights
            if _%50 == 0:
                logging.info("Training Loss" + str(loss_LM.item()))
                if _%500 == 0:

                    #logging.info("Saving " + str(inter_loss) + ' ' + str(metric.compute()))
                    save_metrics(round(inter_loss / 500, 2),
                                 metric.compute()['accuracy'],
                                 checkpoint_path.replace('/checkpoint.pt','_saved'))
                    inter_loss = 0

            epoch_loss += float(loss)

        #print('train accuracy: ', metric.compute())
        msg = "Epoch: {}, Loss: {}, Train accuracy: {}".format(epoch, epoch_loss, metric.compute())
        logging.info(msg)
        epoch += 1
        epoch_loss = 0

        # saving at the end of each epoch
        commit_state(modelCLS, optimizer, rng, epoch, epoch_loss, checkpoint_path)
        save_weights(modelCLS, epoch, save_dir)

        # loading dataframe for the new epoch
        df = pd.read_csv('./abstract_df/df_train_'+str(epoch)+'.csv')
        MAX_LEN = 512
        SUMMARY_LEN = 40
        batch_size = 2
        dataset = abstr_dataset.AbstractsDataset(df, tokenizer, source_len=MAX_LEN, summ_len=SUMMARY_LEN)
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=4)

    return modelCLS




def main(args):

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

    logging.info("starting training script, num-gpu {} {} ".format(args.num_gpu, torch.cuda.device_count()) )
    torch.random.manual_seed(seed)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")

    # check if the checkpoint exists and try to resume from the last checkpoint
    # if you are saving for every epoch, you can skip the part about
    # saving and loading the dataloader state.
    modelCLS, tokenizer, optimizer, dataloader, epoch, cur_loss= \
        create_or_restore_training_state(batch_size=args.batch_size,
                                         mpath=args.mpath,
                                         use_encoder_features=(args.use_encoder_features==1),
                                         checkpoint_path=checkpoint_path,
                                         lr=args.lr)

    # now we can start the training loop
    if args.num_gpu == 1:
        modelCLS = train(modelCLS, tokenizer, optimizer, dataloader, epoch, cur_loss,
                         checkpoint_path, args.num_epoch, args.coef_cls, args.coef_lm)
    else:
        modelCLS = train_with_accelerator(modelCLS, tokenizer, optimizer, dataloader, epoch, cur_loss,
                                          checkpoint_path, args.num_epoch, args.coef_cls, args.coef_lm)



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


    # parser.add_argument(
    #     '--dataset_df_path',
    #     type=str,
    #     help='path to dataframe with abstracts / titles / labels',
    #     required=True,
    # )

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
        '--num_gpu',
        type=int,
        help='Number of GPUs in the job',
        default=1,
    )

    parser.add_argument(
        '--coef_cls',
        type=float,
        help='Weighting coefficient for classification loss',
        default=1.0,
    )

    parser.add_argument(
        '--coef_lm',
        type=float,
        help='Weighting coefficient for language modeling loss',
        default=0.5,
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=1e-5,
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

    # parser.add_argument(
    #     '--checkpoint_interval',
    #     type=int,
    #     help='period to take checkpoints in seconds',
    #     default=500
    # )

    parser.add_argument(
        '--log_file',
        type=str,
        help='specify the location of the output directory, default stdout',
        default=None
    )

    main(parser.parse_args())
