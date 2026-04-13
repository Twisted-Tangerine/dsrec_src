import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter 
from src.trainers.trainer import Trainer
from src.models.backbone import SASRec, Bert4Rec, GRU4Rec
from src.models.dsrec import DSRecSASRec, DSRecBert4Rec, DSRecGRU4Rec
from src.utils.metrics import metric_report, metric_len_report, metric_pop_report
from src.utils.metrics import metric_len_5group, metric_pop_5group
from src.utils.helpers import get_n_params, record_csv


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):
        super().__init__(args, logger, writer, device, generator)

        # Frequency-aware weight computation based on Beta (DSRec variant)
        self.item_weights = None
        beta = getattr(self.args, 'beta', 0.0)  # Beta coefficient (was `freq_beta` in the original paper)

        # Only compute weights when |beta| > 0; otherwise fall back to uniform (None)
        if abs(beta) > 1e-5:
            self.logger.info(f"Calculating Frequency-Aware Weights with Beta = {beta}...")
            
            # 1) Count item frequency
            item_counts = Counter()
            total_batches = len(self.train_loader)
            count_debug_limit = 5  # only log first 5 batches for quick sanity-check
            # More efficient counting (iterate only if dataset hasn’t cached counts)
            for i, batch in enumerate(tqdm(self.train_loader, desc='Scanning Dataset', leave=False)):
                try:
                    # Assume positive items are at batch[1]; adjust if your dataset differs
                    # Typical batch: (seq, pos, neg, positions) or (user, seq, ...)
                    # Ensure we grab correct `pos`; Generator usually outputs targets at batch[1]
                    pos_batch = batch[1].flatten().tolist()
                    
                    # Drop padding tokens (0)
                    valid_pos = [p for p in pos_batch if p != 0]
                    item_counts.update(valid_pos)
                    
                    if i < count_debug_limit:
                        # Quick sanity-check: is this an item id?
                        sample_id = valid_pos[0] if valid_pos else 0
                        if sample_id > self.item_num:
                            self.logger.warning(f"Found item_id {sample_id} > item_num {self.item_num} in batch {i}!")
                except Exception as e:
                    self.logger.error(f"Error counting items in batch {i}: {str(e)}")
                    break

            unique_items_found = len(item_counts)
            self.logger.info(f"[STATS] Unique items found in training: {unique_items_found} / {self.item_num}")
            
            if unique_items_found > 0:
                # Print Top-5 and Tail-5
                most_common = item_counts.most_common(5)
                least_common = item_counts.most_common()[:-6:-1]
                
                self.logger.info("\n" + "="*40)
                self.logger.info("   TOP-5 Frequent Items (Check logic)")
                self.logger.info("="*40)
                for iid, freq in most_common:
                    self.logger.info(f"   Item ID: {iid:<6} | Count: {freq}")
                
                self.logger.info("-" * 40)
                self.logger.info("   TAIL-5 Rare Items (Check logic)")
                self.logger.info("-" * 40)
                for iid, freq in least_common:
                    self.logger.info(f"   Item ID: {iid:<6} | Count: {freq}")
                self.logger.info("="*40 + "\n")
            
            # 2) Compute log-scaled raw weights (long-tail smoothing)
            raw_weights = np.zeros(self.item_num + 1, dtype=float)
            if len(item_counts) > 0:
                for iid, cnt in item_counts.items():
                    raw_weights[iid] = np.log(cnt + 1.0)
                
                # 3) Min–max normalize to [0, 1]
                # Mask of valid items (exclude padding and unseen ids)
                valid_mask = (raw_weights > 0)
                if valid_mask.sum() > 0:
                    w_min = raw_weights[valid_mask].min()
                    w_max = raw_weights[valid_mask].max()
                    if w_max > w_min:
                        raw_weights[valid_mask] = (raw_weights[valid_mask] - w_min) / (w_max - w_min)
                    else:
                        raw_weights[valid_mask] = 0.5
                    
                    offset = (raw_weights[valid_mask] - 0.5) * 2.0
                    final_weights = np.ones(self.item_num + 1, dtype=float)
                    final_weights[valid_mask] = 1.0 + beta * offset
                    final_weights = np.clip(final_weights, 0.01, 3.0) 
                    
                    # Force Mean = 1.0 for Stability
                    current_mean = final_weights[valid_mask].mean()
                    final_weights[valid_mask] /= (current_mean + 1e-8)
                    final_weights[0] = 0.0
                    
                    self.item_weights = torch.tensor(final_weights, dtype=torch.float32).to(self.device)
                    
                    v_w = final_weights[valid_mask]
                    self.logger.info(f"⚖️ [WEIGHTS] Beta={beta}")
                    self.logger.info(f"   Min: {v_w.min():.4f} | Max: {v_w.max():.4f} | Mean: {v_w.mean():.4f} (Target: 1.0)")
                    
                    if beta > 0 and len(most_common) > 0:
                        top_id = most_common[0][0]
                        if final_weights[top_id] < 1.0:
                            self.logger.warning(f"[WARN] Beta > 0 but Top Item {top_id} has weight {final_weights[top_id]:.4f} < 1.0! Check Normalization.")
                else:
                    self.logger.warning("[WARN] Valid mask sum is 0. Something is wrong with counting.")
            else:
                self.logger.warning("[WARN] Item counts are empty! Fallback to None weights.")
        else:
            self.logger.info("[INIT] Beta is 0. Using Standard DSRec (Uniform Weights).")

        self.logger.info('Loading Model: ' + args.model_name)
        self._create_model()
        self.logger.info('# of model parameters: ' + str(get_n_params(self.model)))
        
        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

    def _create_model(self):
        kwargs = {}
        if "dsrec" in self.args.model_name:
            kwargs['item_weights'] = self.item_weights

        '''create your model'''
        if self.args.model_name == 'sasrec':
            self.model = SASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == 'bert4rec':
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == 'gru4rec':
            self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "dsrec_sasrec":
            self.model = DSRecSASRec(self.user_num, self.item_num, self.device, self.args, **kwargs)
        elif self.args.model_name == "dsrec_bert4rec":
            self.model = DSRecBert4Rec(self.user_num, self.item_num, self.device, self.args, **kwargs)
        elif self.args.model_name == "dsrec_gru4rec":
            self.model = DSRecGRU4Rec(self.user_num, self.item_num, self.device, self.args, **kwargs)
        else:
            raise ValueError
        
        self.model.to(self.device)

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        for batch in self.train_loader:
            batch = tuple(t.to(self.device) for t in batch)
            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)

            self.optimizer.zero_grad()
            loss = self.model(**inputs)
            loss.backward()
            self.optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1
            train_end = time.time()
            train_time.append(train_end-train_start)

        avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
        self.logger.info(f"Epoch {epoch}: Avg training loss = {avg_loss:.4f}")
        self.writer.add_scalar('train/loss', avg_loss, epoch)

        avg_train_time = float(np.mean(train_time)) if len(train_time) > 0 else 0.0
        return avg_train_time

    def eval(self, epoch=0, test=False, model=None):
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'), weights_only=False)
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
            eval_model = self.model
        
        else:       
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in test_loader:

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("Item Performance:")
        
        tail_ndcg = res_pop_dict.get('Tail NDCG@10', 0)
        tail_hr = res_pop_dict.get('Tail HR@10', 0)
        self.logger.info('\t Tail NDCG@10: %.5f' % tail_ndcg)
        self.logger.info('\t Tail HR@10: %.5f' % tail_hr)

        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_pop_dict} if not test else {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict

    def save_user_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'), weights_only=False)
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        pickle.dump(user_emb, open("./data/{}/user_id_embeddings.pkl".format(self.args.dataset), "wb"))

    def save_item_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'), weights_only=False)
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        all_index = torch.arange(start=1, end=self.item_num+1).to(self.device)
        item_emb = self.model._get_embedding(all_index)
        item_emb = item_emb.detach().cpu().numpy()
        pickle.dump(item_emb, open("./data/{}/item_id_embeddings.pkl".format(self.args.dataset), "wb"))

    def test_group(self):
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [5, 10, 20, 40])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))

        return res_dict