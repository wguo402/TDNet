from torch.utils.data import DataLoader
from datasets.stnet_dataset import TrainDataset, TestDataset, TestDataset_WOD

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_dataset(opts, partition, shuffle=False):
    loader, db = None, None
    
    if opts.use_tiny:
        split = "Tiny_" + partition
    else:
        split = "Full_" + partition
    
    
    if partition in ["Train", "Valid"]:
        db = TrainDataset(opts, split)#is very import in opts
        loader = DataLoader(db, batch_size=opts.batch_size, shuffle=shuffle, num_workers=opts.n_workers, pin_memory=True)
        if partition in ["Train"]:
            train_iter=get_data_iterator(loader)
        else:
            train_iter=0
    else:
        # Test dataset
        if opts.which_dataset.upper() in ['KITTI', 'NUSCENES']:
            db = TestDataset(opts, split)
            loader = DataLoader(db, batch_size=1, shuffle=shuffle, num_workers=opts.n_workers, pin_memory=True, collate_fn=lambda x: x)
            train_iter=0
        else:
            # waymo test
            db = TestDataset_WOD(opts, pc_type='raw_pc')
            train_iter=0

    return loader, db,train_iter