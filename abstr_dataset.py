from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# with a very large training data, an epoch may take longer than your permitted
# time slice. Therefore, you might want to save checkpoint after certain
# iterations within an epoch. However, if you are randomizing the batches for
# each epoch, you need to save the random state in case of preemption.,
# PyTorch currently does not have the out-of-box solution, but you could write
# an customized sampler to do so.
class StatefulSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data = data_source
        self.shuffle = shuffle

        # initial dataloader index
        self.init_index()

    def init_index(self):

        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        else:
            self.indices = torch.arange(len(self.data))

        self.data_counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.data_counter == len(self.data):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size
        return {
                'indices': self.indices,
                'data_counter': self.data_counter - prefetched_num,
            }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']


class DistributedSaveableSampler(DistributedSampler):
    """Just like with the case with
        torch.utils.data.distributed.DistributedSampler you *MUST* call
        self.set_epoch(epoch:int) to ensure all replicates use the same
        random shuffling within each epoch if shuffle is True
    """

    def __init__(self, *args, force_synchronization=False, **kwargs):
        """
        Arguments:
            force_synchronization (boolean, optional): If it's true then after
                each yield we will force a synchronization so each process'
                _curr_idx will be the same, this guarantees correctness of the
                save in case there is no synchronization during training, but
                comes at a performance cost
            For the rest of the arguments please see:
                https://pytorch.org/docs/1.7.1/data.html?highlight=distributed%20sampler#torch.utils.data.distributed.DistributedSampler

        """
        super().__init__(*args, **kwargs)
        self._curr_idx = 0
        self.force_synchronization = force_synchronization

    def __iter__(self):
        """Logic modified from
            https://pytorch.org/docs/1.7.1/_modules/torch/utils/data/distributed.html#DistributedSampler
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset),
                                     generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        while self._curr_idx + self.rank < self.total_size:
            to_yield = self.rank + self._curr_idx

            # we need to increment this before the yield because
            # there might be a save or preemption while we are yielding
            # so we must increment it before to save the right index
            self._curr_idx += self.num_replicas

            yield to_yield

            if self.force_synchronization:
                dist.barrier()
        self._curr_idx = 0

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = (
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) *
                    batch_size)

        return {
            "index": self._curr_idx - (prefetched_num * self.num_replicas),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self._curr_idx = state_dict["index"]
        self.epoch = state_dict["epoch"]



# Dataset class to generate abstract, title and journal label from pandas dataframe
class AbstractsDataset(Dataset):

    def __init__(self,
                 df,
                 tokenizer,
                 source_len = 512,
                 summ_len = 35,
                 title_column = '1',
                 abstr_column = '4',
                 label_column = 'label',
                 do_summarization = True):
        self.tokenizer = tokenizer
        self.my_data = df
        self.source_len = source_len
        self.summ_len = summ_len
        self.titles = self.my_data[title_column]
        self.abstracts = self.my_data[abstr_column]
        self.labels = self.my_data[label_column]
        self.do_summarization = do_summarization

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        label = int(self.labels[index]) # journal label

        text = str(self.abstracts[index]) # original abstract
        text = ' '.join(text.split())

        if self.do_summarization:
            sum_text = str(self.abstracts[index]) # abstract with "summarize" prefix
            sum_text = ' '.join(sum_text.split())

        title = str(self.titles[index]) # corresponding title
        title = ' '.join(title.split())

        joint = title + self.tokenizer.sep_token + text

        text = self.tokenizer.batch_encode_plus([text],
                                                 max_length = self.source_len,
                                                 pad_to_max_length=True,
                                                 truncation=True,
                                                 return_tensors='pt')
        if self.do_summarization:
            source = self.tokenizer.batch_encode_plus(['summarize: '+sum_text],
                                                  max_length = self.source_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  return_tensors='pt')

            target = self.tokenizer.batch_encode_plus([title],
                                                      max_length = self.summ_len,
                                                      pad_to_max_length=True,
                                                      truncation=True,
                                                      return_tensors='pt')
        else:
            joint_abstr_title = self.tokenizer.batch_encode_plus([joint],
                                                                 max_length = self.source_len,
                                                                 pad_to_max_length=True,
                                                                 truncation=True,
                                                                 return_tensors='pt')

        cls_ids = text['input_ids'].squeeze()
        cls_mask = text['attention_mask'].squeeze()
        if self.do_summarization:
            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
            target_ids = target['input_ids'].squeeze()
            target_mask = target['attention_mask'].squeeze()
        else:
            # joint title + abstract input
            joint_abstr_title_ids = joint_abstr_title['input_ids'].squeeze()
            joint_abstr_title_mask = joint_abstr_title['attention_mask'].squeeze()


        if self.do_summarization:
            return {
                'input_ids': cls_ids.to(dtype=torch.long),
                'attention_mask': cls_mask.to(dtype=torch.long),
                'label': torch.tensor(label),
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
                'target_ids': target_ids.to(dtype=torch.long),
                'target_ids_y': target_ids.to(dtype=torch.long)
            }

        else:
            return {
                'input_ids': cls_ids.to(dtype=torch.long),
                'attention_mask': cls_mask.to(dtype=torch.long),
                'label': torch.tensor(label),
                'joint_abstr_title_ids': joint_abstr_title_ids.to(dtype=torch.long),
                'joint_abstr_title_mask': joint_abstr_title_mask.to(dtype=torch.long)
            }
