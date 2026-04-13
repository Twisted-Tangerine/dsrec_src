# Standard library and third-party imports
import os
import numpy as np
import torch
from tqdm import tqdm, trange
from src.utils.earlystop import EarlyStoppingNew

class MultiMetricEarlyStopping():
    """Built-in multi-metric early stopping for long-tail recommendation."""
    
    def __init__(self, patience=7, verbose=False, delta=0.001, path='./checkpoint/', 
                 trace_func=print, overall_metric='NDCG@10', tail_metric='Tail NDCG@10',
                 overall_weight=0.4, tail_weight=0.6, min_tail_improvement=0.002):
        """
        Args:
            patience (int): How long to wait after last time validation performance improved.
            verbose (bool): If True, prints a message for each validation improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
            overall_metric (str): Overall performance metric name.
            tail_metric (str): Tail performance metric name.
            overall_weight (float): Weight for overall metric (0-1).
            tail_weight (float): Weight for tail metric (0-1).
            min_tail_improvement (float): Minimum improvement required for tail metric.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.path = os.path.join(path, "pytorch_model.bin")
        self.trace_func = trace_func
        
        # Multi-metric parameters
        self.overall_metric = overall_metric
        self.tail_metric = tail_metric
        self.overall_weight = overall_weight
        self.tail_weight = tail_weight
        self.min_tail_improvement = min_tail_improvement
        
        # Track individual metrics
        self.best_overall_score = None
        self.best_tail_score = None
        self.overall_scores = []
        self.tail_scores = []

    def _log(self, msg: str):
        if not self.verbose:
            return
        logger_like = getattr(self.trace_func, 'info', None)
        if callable(logger_like):
            logger_like(msg)
        elif callable(self.trace_func):
            self.trace_func(msg)

    def __call__(self, metric_dict, epoch, model, optimizer=None, scheduler=None):
        """
        Args:
            metric_dict (dict): Dictionary containing all metrics
            epoch (int): Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
        """
        overall_score = metric_dict.get(self.overall_metric, 0)
        tail_score = metric_dict.get(self.tail_metric, 0)

        # Store scores for analysis
        self.overall_scores.append(overall_score)
        self.tail_scores.append(tail_score)

        # First epoch: initialize bests and save
        if self.best_score is None:
            # For relative scheme, initialize bests with the first observed scores
            self.best_overall_score = overall_score
            self.best_tail_score = tail_score
            # Weighted relative improvement is defined from the second epoch; use absolute weighted score for init logging
            init_weighted = self.overall_weight * overall_score + self.tail_weight * tail_score
            self.best_score = init_weighted
            self.best_epoch = epoch
            self.save_checkpoint(init_weighted, model, optimizer, scheduler, epoch)
            self._log(f'[MultiMetric] Init - Overall: {overall_score:.5f}, Tail: {tail_score:.5f}, Weighted(abs): {init_weighted:.5f}')
            return

        # Relative improvements against historical bests (percentage improvements)
        eps = 1e-8
        rel_overall = (overall_score - self.best_overall_score) / max(eps, self.best_overall_score)
        rel_tail = (tail_score - self.best_tail_score) / max(eps, self.best_tail_score)
        weighted_rel = self.overall_weight * rel_overall + self.tail_weight * rel_tail

        if (weighted_rel > self.delta) and (rel_tail >= self.min_tail_improvement):
            # Consider it an improvement only if weighted relative gain is sufficient
            # and tail also achieves the minimum relative improvement
            self.best_overall_score = max(self.best_overall_score, overall_score)
            self.best_tail_score = max(self.best_tail_score, tail_score)
            self.best_score = weighted_rel
            self.best_epoch = epoch
            self.save_checkpoint(weighted_rel, model, optimizer, scheduler, epoch)
            self.counter = 0
            self._log(f'[MultiMetric] Improve - rel_overall: {rel_overall:+.5f}, rel_tail: {rel_tail:+.5f}, weighted_rel: {weighted_rel:+.5f}')
        else:
            self.counter += 1
            self._log(f'[MultiMetric] Hold - rel_overall: {rel_overall:+.5f}, rel_tail: {rel_tail:+.5f}, weighted_rel: {weighted_rel:+.5f}, counter: {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                self._log(f'[MultiMetric] Early stopping triggered at epoch {epoch}')

    def save_checkpoint(self, score, model, optimizer, scheduler, epoch):
        """Saves model when performance improves."""
        self._log(f'[MultiMetric] Best weighted score improved to {score:.6f}. Saving model...')
        
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'best_overall_score': self.best_overall_score,
            'best_tail_score': self.best_tail_score,
            'best_weighted_score': self.best_score
        }, self.path)


class Trainer(object):

    def __init__(self, args, logger, writer, device, generator):

        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = device
        self.generator = generator
        self.user_num, self.item_num = generator.get_user_item_num()
        self.start_epoch = 0    # define the start epoch for keepon training

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.train_loader = generator.make_trainloader()
        self.valid_loader = generator.make_evalloader()
        self.test_loader = generator.make_evalloader(test=True)

        # get item pop and user len
        self.item_pop = generator.get_item_pop()
        self.user_len = generator.get_user_len()

        #self.watch_metric = 'NDCG@10'  # use which metric to select model
        self.watch_metric = args.watch_metric

    
    def _create_model(self):
        '''create your model'''
        raise NotImplementedError
    

    def _load_pretrained_model(self):

        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        # filter out required parameters
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # Print: how many parameters are loaded from the checkpoint
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  # load model parameters
        self.optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        self.scheduler.load_state_dict(checkpoint['scheduler']) # load scheduler
        self.start_epoch = checkpoint['epoch']  # load epoch

    
    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr,
                                          weight_decay=self.args.l2,
                                          )

    
    def _set_scheduler(self):

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.args.lr_dc_step,
                                                         gamma=self.args.lr_dc)


    def _set_stopper(self):
        # Choose early stopping strategy based on model type
        # DSRec models: use multi-metric early stopping (Overall + Tail NDCG@10)
        # Base models: use single-metric early stopping (Overall NDCG@10 only)
        is_dsrec_model = self.args.model_name in ["dsrec_sasrec", "dsrec_bert4rec", "dsrec_gru4rec"]
        
        if is_dsrec_model:
            # Multi-metric early stopping for DSRec models
            # This helps prevent overfitting on tail items by balancing overall and tail performance
            self.stopper = MultiMetricEarlyStopping(
                patience=self.args.patience,
                verbose=True,  # Enable verbose logging to track tail performance
                path=self.args.output_dir,
                trace_func=self.logger,
                overall_metric=self.watch_metric,
                tail_metric='Tail NDCG@10',
                overall_weight=0.4,  # 40% weight for overall performance
                tail_weight=0.6,     # 60% weight for tail performance (higher priority)
                min_tail_improvement=0.002  # Minimum tail improvement threshold
            )
            self.logger.info(f"[MultiMetric Early Stop] Enabled with patience={self.args.patience}")
            self.logger.info(f"[MultiMetric Early Stop] Overall weight: 0.4, Tail weight: 0.6")
            self.logger.info(f"[MultiMetric Early Stop] Monitoring: {self.watch_metric} + Tail NDCG@10")
        else:
            # Single-metric early stopping for base models (sasrec, bert4rec, gru4rec)
            self.stopper = EarlyStoppingNew(
                patience=self.args.patience,
                verbose=True,
                delta=0.001,
                path=self.args.output_dir,
                trace_func=self.logger
            )
            self.logger.info(f"[SingleMetric Early Stop] Enabled with patience={self.args.patience}")
            self.logger.info(f"[SingleMetric Early Stop] Monitoring: {self.watch_metric} only")

    def _train_one_epoch(self, epoch):
        raise NotImplementedError
    

    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training"""
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    

    def _prepare_eval_inputs(self, data):
        """Prepare the inputs as a dict for evaluation"""
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs


    def eval(self, epoch=0, test=False):
        raise NotImplementedError


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.num_train_epochs), desc="Epoch"):

            t = self._train_one_epoch(epoch)
            
            train_time.append(t)

            # evluate on validation per 20 epochs
            if (epoch % 1) == 0:
                
                metric_dict = self.eval(epoch=epoch)
                res_list.append(metric_dict)
                #self.scheduler.step()
                # Pass metrics to stopper based on type
                if isinstance(self.stopper, MultiMetricEarlyStopping):
                    # Multi-metric stopper expects a dict
                    self.stopper(metric_dict, epoch, model_to_save, self.optimizer, self.scheduler)
                else:
                    # Single-metric stopper expects a float
                    self.stopper(metric_dict[self.watch_metric], epoch, model_to_save, self.optimizer, self.scheduler)

                if self.stopper.early_stop:
                    self.logger.info(f"[EarlyStop] stopping at epoch {epoch}, best epoch: {self.stopper.best_epoch}")
                    break
        
        best_epoch = self.stopper.best_epoch
        best_res = res_list[best_epoch - self.start_epoch]
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)
        self.logger.info('The best results are NDCG@10: %.5f, HR@10: %.5f' %
                    (best_res['NDCG@10'], best_res['HR@10']))
        
        res = self.eval(test=True)

        return res, best_epoch
    


    def test(self):
        """Do test directly. Set the output dir as the path that save the checkpoint"""
        res = self.eval(test=True)

        return res, -1



    def get_model(self):

        return self.model

    
    def get_model_param_num(self):

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_num = total_num - trainable_num

        return freeze_num, trainable_num
