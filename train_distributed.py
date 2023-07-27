# pylint: skip-file
import argparse
from tools.train_test import train
import torch
import torch.nn as nn
from models.cnn_rnn import RecurrentResnet, RecurrentResnetWPacking
from dataset.dataloader import BatchPadV2, load_data
from tools.distributed import init_distributed_mode
from tools.logs import get_logger

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr-decay-milestones", default="30,50", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
parser.add_argument('--output-dir', default='./output', help='path where to save')
parser.add_argument("--seed", type=int, default=None)

parser.add_argument('--gpu', default='0', help='num of gpu to be used')
parser.add_argument("--discription", type=str, default=None, help="discription of this experiment")
parser.add_argument("--restored_model", type=str, default=None, help="path of restored model, default None")

# distributed setting
parser.add_argument("--distributed", type=bool, default=True, help="if use distributed training")

args = parser.parse_args()
args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

### init distributed ###
init_distributed_mode(args)

### get logger ###
log_file = f"{args.output_dir}/{args.discription}.txt"
logger = get_logger(name='train', output=log_file, color=True)
### criterion ###
criterion  = nn.CrossEntropyLoss()
### get model ###
if args.restored_model is not None:
    model = RecurrentResnetWPacking()
    state_dict = torch.load(args.restored_model)
    model.load_state_dict(state_dict)
else:
    model = RecurrentResnetWPacking()
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
else:
    model = model.to(args.device)

logger.info("get model successfully")

### get dataloader ###
data_dir = '/home/public_datasets/FERV39k'
csv_train = '/home/public_datasets/FERV39k/FERV39k/4_setups/All_scenes/train_All.csv'
csv_test = '/home/public_datasets/FERV39k/FERV39k/4_setups/All_scenes/test_All.csv'
train_data, test_data, train_sampler, test_sampler = load_data(data_dir, csv_train, csv_test, args)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           collate_fn=BatchPadV2(),
                                           drop_last=True,
                                           sampler=train_sampler,
                                           num_workers=16)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=args.batch_size,
                                          collate_fn=BatchPadV2(),
                                          drop_last=True,
                                          sampler=test_sampler,
                                          num_workers=16)
logger.info("get dataloader successfully")

#TODO add count_ops_and_params

### train ###
logger.info("start training")
train(model, train_loader, test_loader, criterion, logger, args)