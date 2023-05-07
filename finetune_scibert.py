import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from accelerate import Accelerator

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_metric

import abstr_dataset
import logging, os, argparse
import transformers, datasets
from accelerate import Accelerator


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


def save_weights(accelerator, model, save_dir, epoch):
    filename = os.path.join(save_dir, 'weights_'+str(epoch)+'.pt')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), filename)



def training_function(args):
    logging.info("Launching")
    # Initialize accelerator
    accelerator = Accelerator()

    logging.info("Num processes in accelerator = {}".format(str(accelerator.num_processes)))

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    mpath = 'allenai/scibert_scivocab_uncased'
    SciBertCls = BertForSequenceClassification.from_pretrained(mpath, num_labels = 2734)
    tokenizer = AutoTokenizer.from_pretrained(mpath)

    save_dir = args.checkpoint_dir
    if accelerator.is_main_process and not os.path.exists(save_dir): os.mkdir(save_dir)
    if os.path.exists(save_dir+'epoch.npy'):
        epoch = int(np.load(save_dir+'epoch.npy'))
    else:
        epoch = 0

    df = pd.read_csv('./abstracts_scibert_df/df_'+str(epoch%10)+'.csv') # read df corresponding to the current epoch
    train_params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4,
    }

    ds_params = {
        'title_column': 'title',
        'abstr_column': 'abstr',
        'label_column': 'label_journal',
        'do_summarization': False,
    }

    ds = abstr_dataset.AbstractsDataset(df, tokenizer, **ds_params)
    train_dataloader = DataLoader(ds, **train_params)

    filename = os.path.join(save_dir, 'weights_'+str(epoch)+'.pt')
    if os.path.exists(filename) and accelerator.is_main_process:
        logging.info('epoch =  '+str(epoch))
        logging.info('loading '+filename)
        SciBertCls.load_state_dict(torch.load(filename))
    else:
        logging.info('no ckpt detected')

    # Instantiate optimizer
    optimizer = AdamW(params=SciBertCls.parameters(), lr=args.lr)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    SciBertCls, optimizer, train_dataloader = accelerator.prepare(
        SciBertCls, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
    # process to avoid having 8 progress bars.
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)),
                        initial = epoch*len(train_dataloader),
                        disable=not accelerator.is_main_process)

    inter_loss = 0 # intermediate train loss to keep track

    if args.use_title==1:
        # joint title + abstr input
        input_ids_code, attention_mask_code = 'joint_abstr_title_ids', 'joint_abstr_title_mask'
    else:
        input_ids_code, attention_mask_code = 'input_ids', 'attention_mask'

    while epoch < num_epochs:
        SciBertCls.train()
        metric = load_metric('accuracy')

        logging.info('epoch')
        for _, batch in enumerate(train_dataloader):
            # CLASSIFICATION
            # outputs = SciBertCls(input_ids=batch['input_ids'],#.to(device),
            #                      attention_mask=batch['attention_mask'],#.to(device),
            #                      labels=batch['label']#.to(device)
            #                     )
            outputs = SciBertCls(input_ids=batch[input_ids_code],
                                 attention_mask=batch[attention_mask_code],
                                 labels=batch['label']
                                )

            # TOTAL LOSS
            loss = outputs.loss
            inter_loss += int(loss)

            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions,
                             references=batch["label"])
            progress_bar.update(1)

            # save train metrics
            if _%1000 == 0 and _>0 and accelerator.is_main_process:
                save_metrics(round(inter_loss / 1000, 2),
                             metric.compute()['accuracy'],
                             save_dir)
                inter_loss = 0

        # saving model at the end of the epoch
        logging.info('saving weights' )
        #save_weights(accelerator, SciBertCls, '/scratch/hdd001/home/anastasia/scibert_finetune/', epoch)
        save_weights(accelerator, SciBertCls, args.checkpoint_dir, epoch)

        # new epoch
        epoch += 1
        np.save(save_dir+'epoch', epoch) # we know where to start from next time
        logging.info('reading df '+str(epoch%10))
        df = pd.read_csv('./abstracts_scibert_df/df_'+str(epoch%10)+'.csv') # read df corresponding to the current epoch
        #df = df.iloc[:50]
        print(df.head())
        ds = abstr_dataset.AbstractsDataset(df, tokenizer, **ds_params)
        train_dataloader = DataLoader(ds, **train_params)
        train_dataloader = accelerator.prepare(train_dataloader)



def main(args):

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

    #logging.info("starting training script")
    logging.info("starting training script, num-gpu {} ".format(torch.cuda.device_count()) )
    #torch.random.manual_seed(seed)

    training_function(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script that performs checkpointing for PyTorch'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        help='specify the location of the output directory, default stdout',
        default=None
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='number of epochs to run',
        default=50,
    )

    parser.add_argument(
        '--use_title',
        type=int,
        help='whether to use title as an extra model input',
        default=0,
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=5e-6,
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='path to save and look for the checkpoint file',
        required=True,
    )


    main(parser.parse_args())
