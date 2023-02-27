import time
import os
import yaml

config = yaml.load(open('./config_crack.yml'), Loader=yaml.FullLoader)

class Visualizer():
    def __init__(self, isTrain=False):
        self.log_name = os.path.join('./checkpoints', config['loss_filename'])
        self.isTrain = isTrain
        if self.isTrain:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training loss (%s) ================\n' % now)
        else:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing begin (%s) ================\n' % now)

    def print_current_losses(self, epoch=0, iters=0, loss=0.,  lr=0., isVal=False):
        """print current losses on console; also save the losses to the disk
        """
        if not isVal: # train
            message = '(epoch: %d, iters: %d) mean_loss: %6f lr: %6f' % (epoch, iters, loss, lr)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        elif isVal: # val
            message = 'validation on epoch>> %d, mean tloss>> %6f ' % (epoch, loss)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('val_mode:%s\n' % message)  # save the message

    def print_end(self, best=0, best_val_loss=0.):
        message = 'best model appear in epoch%d and best val_loss is %6f' % (best, best_val_loss)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Training End (%s) ================\n' % end_now)

    def print_val(self, tn, fp, fn, tp, precision, recll, f1):
        message = 'TN=%d, FP= %d, FN=%d, TP=%d\nprecision:%6f, recall:%6f, F1_score:%6f' % (tn, fp, fn, tp, precision, recll, f1)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Testing End (%s) ================\n' % end_now)
                

