class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return '/mnt/xwj/datasets/Cityscapes/'  # folder that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
