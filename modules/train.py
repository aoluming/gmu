import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
import pdb
import numpy as np
from .metrics import eval_result
from sklearn import metrics
import torchvision.models as models
class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

class gmuTrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None, logger=None,  writer=None,target_names=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.target_names=target_names
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.vgg = models.vgg19(pretrained=True).cuda()
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.lossfn=torch.nn.BCELoss()
        self.before_gmu_train()
        # pdb.set_trace()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    loss, logits, labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # if self.step==250:
                    #     pdb.set_trace()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    loss, logits, labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    true_labels.extend(labels.detach().cpu().tolist())
                    pred_labels.extend(logits.detach().cpu().tolist())
                    pbar.update()
                results=self.report_performance(true_labels,pred_labels,0.5)
                # evaluate done
                pbar.close()
                # if self.writer:
                #     self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)    # tensorbordx
                #     self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)    # tensorbordx
                #     self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx

                # self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}."\
                #             .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if results['macro'][3] >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = results['macro'][3] # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
                

        self.model.train()
    def report_performance(self,y_true, y_prob, threshold, print_results=True, multilabel=True):
        # y_true=y_true.cpu().numpy()
        # y_prob=y_prob.cpu().numpy()
        y_true=np.array(y_true)
        y_prob=np.array(y_prob)
        y_pred = y_prob > threshold

        # print("y_pred",y_pred)
        # y_true=y_true[0]
        # y_prob=y_prob[0]
        results = {}
        averages = ('micro', 'macro', 'weighted', 'samples')
        if multilabel:
            acc = metrics.accuracy_score(y_true, y_pred)
        else:
            acc = metrics.accuracy_score(
                y_true.argmax(axis=1), y_prob.argmax(axis=1))
        # print(acc)
        # pdb.set_trace()
       
        for average in averages:
            results[average] = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)[:3] + (
                # metrics.roc_auc_score(y_true, y_prob, average=average),
                metrics.hamming_loss(y_true, y_pred),
                acc)

        if print_results:
            print('average\tprecisi\trecall\tf_score\thamming\taccuracy')
            for avg, vals in results.items():
                print('{0:.7}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}\t{5:0.3f}'.format(
                    avg, *vals))
            
            print(metrics.classification_report(
                y_true, y_pred, target_names=self.target_names))
        return results
    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    loss, logits, labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    # pdb.set_trace()
                    true_labels.extend(labels.detach().cpu().tolist())
                    pred_labels.extend(logits.detach().cpu().tolist())
                    pbar.update()
                results=self.report_performance(true_labels,pred_labels,0.5)
                # evaluate done
                pbar.close()
                # if self.writer:
                #     self.writer.add_scalar(tag='test_acc', scalar_value=acc)    # tensorbordx
                #     self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)    # tensorbordx
                #     self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))    # tensorbordx
                total_loss = 0
                # self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))
                    
        self.model.train()
        
    def _step(self, batch, mode="train"):
        if mode != "predict":

            image,text,label=batch
            text=text.to(self.args.device)
            image=image.to(self.args.device)
            label=label.to(self.args.device)
            a=self.vgg.features(image)#使用feature获取特征层（卷积层）的特征；输出特征维度为【1，512，4，4】
            b=self.vgg.avgpool(a)#使用vgg定义的池化操作；输出特征维度为【1，512，7，7】
            b=torch.flatten(b,1)#将特征变成一维度；输出特征维度为【1，25088】
            img=self.vgg.classifier[:4](b)#使用分类层的的第一层，当然可以选择数；输出特征维度为【1，4096】



            # inputs=inputs.cuda()
            # pdb.set_trace()
            outputs,z=self.model(img,text)
            
            outputs=outputs.to(torch.float32)
            label=label.to(torch.float32)
            loss=self.lossfn(outputs, label)
            # if self.args.use_prompt:
            #     input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch
            # else:
            #     images, aux_imgs = None, None
            #     input_ids, token_type_ids, attention_mask, labels= batch
            # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs)
            return loss,outputs, label

    def before_gmu_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        # self.optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.args.lr,momentum =0.9)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


    def before_multimodal_train(self):
        optimizer_grouped_parameters = []
        params = {'lr':self.args.lr, 'weight_decay':0}
        params['params'] = []
        #params2 = {'lr':self.args.lr*100, 'weight_decay':1e-2}
        params2 = {'lr':self.args.lr, 'weight_decay':0}
        params2['params'] = []
        for p in self.vgg.parameters():
            p.requires_grad=False
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.require_grad = False
                #params['params'].append(param)
                
        optimizer_grouped_parameters.append(params)
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                
                params2['params'].append(param)
        optimizer_grouped_parameters.append(params2)

        # params = {'lr':self.args.lr, 'weight_decay':1e-2}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'encoder_conv' in name or 'gates' in name:
        #         params['params'].append(param)
        # optimizer_grouped_parameters.append(params)

        # # freeze resnet
        # for name, param in self.model.named_parameters():
        #     if 'image_model' in name:
        #         param.require_grad = False
        # self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.optimizer = optim.Adam(optimizer_grouped_parameters)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


