r""" Logging during training/testing """
import datetime
import logging
import os
import sys

from tensorboardX import SummaryWriter
import torch


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training, tbd, resume):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if resume: logpath += '_resume'
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        os.makedirs(cls.logpath)

        if len(logging.root.handlers) > 0:
            logging.root.removeHandler(logging.root.handlers[0])
        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S',
                            force=True)

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.flush = sys.stdout.flush
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        cls.logger = logging.getLogger('')
        cls.logger.addHandler(console)

        # Tensorboard writer
        if tbd:
            cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        cls.logger.info('\n:============= Geometric Assembly =============')
        for arg_key in args.__dict__:
            cls.logger.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        cls.logger.info(':================================================\n')
        cls.ckp_metric = {}

    @classmethod
    def log_param(cls, model):

        def _count_model_param(model):
            n_param = 0
            for k in model.state_dict().keys():
                n_param += model.state_dict()[k].reshape(-1).contiguous().size(0)
            return n_param

        Logger.info('Total # model param: %d' % _count_model_param(model))
        Logger.info('Total # backbone param: %d' % _count_model_param(model.backbone))

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        cls.logger.info(msg)
        cls.logger.handlers[0].flush()

    @classmethod
    def save_multiple_ckp(cls, model, optimizer, epoch, eval_result):

        for metric in eval_result:
            if cls.ckp_metric.get(metric) is None:
                cls.ckp_metric[metric] = float('inf')

            value = eval_result[metric]
            if cls.ckp_metric[metric] > value:
                cls.ckp_metric[metric] = value
                cls.save_ckp(model, optimizer, epoch, value, ckp_name=metric)

    @classmethod
    def save_ckp(cls, model, optimizer, epoch, value, ckp_name):
        ckp_path = os.path.join(cls.logpath, 'best_model_%s.pt' % ckp_name)
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(checkpoint, ckp_path)

        value = '' if value is None else '%.4f' % value
        cls.info('Model saved @%d w/ %s. %s\n' % (epoch, ckp_name, value))

    @classmethod
    def load_ckp(cls, model, optimizer, ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    @classmethod
    def write_process(cls, idx, datalen, average_meter, epoch, write_idx):
        if (idx + 1) % write_idx != 0:
            return

        eval_stat, loss_stat = average_meter.get_result()

        msg = '[E%2d]' % epoch if epoch != -1 else ''
        msg += '[%5d/%5d]' % (idx+1, datalen)

        for k in loss_stat:
            msg += f' %4.2f ({k})|' % loss_stat[k].mean().item()
        msg += '@|'
        for k in eval_stat:
            msg += f' %4.2f ({k})|' % eval_stat[k]['sample_mean'].mean().item()
        Logger.info(msg)

    @classmethod
    def write_result(cls, average_meter, epoch, training, tbd):

        eval_stat, loss_stat = average_meter.get_result()

        msg = '[E%2d] ' % epoch if epoch != -1 else ''
        msg += '\n %s results: \n' % ('Trainig' if training else 'Testing')

        pre = 'trn' if training else 'val'
        for k in loss_stat:
            value = loss_stat[k].mean().item()
            msg += '%s: %4.3f \n' % (k, value)
            if tbd: Logger.tbd_writer.add_scalars('data/%s_%s' % (pre, k), {k: value}, epoch)

        cls_order = ['BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', \
                     'Plate', 'Spoon', 'Teacup', 'ToyFigure', 'WineBottle', \
                     'Bottle', 'Cookie', 'DrinkBottle', 'Mirror', 'PillBottle', \
                     'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass', \
                     'sample_mean', 'class_mean']

        header = '%12s' % 'Obj class: '
        for k in eval_stat:
            metric_msg = '%10s: ' % k
            for obj_class in cls_order:
                if eval_stat[k].get(obj_class) is None: continue
                if header[-1] != '\n':
                    header += '%6s |' % obj_class[:5]
                value = eval_stat[k][obj_class].mean().item()
                metric_msg += ' %5.2f |' % value

                if tbd and obj_class == 'sample_mean':
                    Logger.tbd_writer.add_scalars('data/%s_%s' % (pre, k), {k: value}, epoch)

            if header[-1] != '\n':
                header += '\n'
                msg += header
            msg += metric_msg
            msg += '\n'

        msg += '\n\n'
        Logger.info(msg)
        if tbd:
            Logger.tbd_writer.flush()


class AverageMeter:

    def __init__(self, dataset):
        self.dataset = dataset
        self.initialized = False
        self.eval_buf = {}
        self.loss_buf = {}

    def update(self, eval_result, loss):

        for k in eval_result.keys():
            if not self.initialized: self.eval_buf[k] = {'sample_mean': []}
            self.eval_buf[k]['sample_mean'].append(eval_result[k].clone().detach())

        for k in loss.keys():
            if not self.initialized: self.loss_buf[k] = []
            self.loss_buf[k].append(loss[k].clone().detach())

        self.initialized = True

    def update_test(self, obj_class, eval_result, loss):

        for k in eval_result.keys():
            if self.eval_buf.get(k) is None:
                self.eval_buf[k] = {}
            if self.eval_buf[k].get(obj_class) is None:
                self.eval_buf[k][obj_class] = []
            self.eval_buf[k][obj_class].append(eval_result[k])

        for k in loss.keys():
            if self.loss_buf.get(k) is None:
                self.loss_buf[k] = []
            self.loss_buf[k].append(loss[k])

    def get_result(self):

        eval_result = {}
        for k in self.eval_buf:
            eval_result[k] = {}
            eval_result_class = []
            eval_result_sample = []
            for obj_class in self.eval_buf[k]:
                class_mean = torch.stack(self.eval_buf[k][obj_class]).mean(dim=0)
                eval_result[k][obj_class] = class_mean
                eval_result_class += [class_mean]
                eval_result_sample += self.eval_buf[k][obj_class]
            eval_result[k]['sample_mean'] = torch.stack(eval_result_sample).mean(dim=0)
            eval_result[k]['class_mean'] = torch.stack(eval_result_class).mean(dim=0)

        loss_result = {}
        for k in self.loss_buf:
            loss_result[k] = torch.stack(self.loss_buf[k]).mean()

        return eval_result, loss_result

    def get_result_of(self, metric):

        eval_result = self.get_result()[0]

        eval_result_of = {}
        for m in metric:
            eval_result_of[m] = eval_result[m]['sample_mean'].mean().item()

        return eval_result_of