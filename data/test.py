
from .utils import MSADataSet
import torch



if __name__=='__main__':
    data_path='../dataset/'
    msa_dataset=MSADataSet(data_path,num_msa_files=100)
    val_size=20
    train_size=len(msa_dataset)-val_size
    train_set,val_set=torch.utils.data.random_split(msa_dataset,[train_size,val_size])
    torch.save(train_set,'../test_1.bin')