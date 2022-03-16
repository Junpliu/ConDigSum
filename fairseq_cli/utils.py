import json
import random, copy, os, re
import pickle as pkl
import numpy as np
from rouge import Rouge
import torch
from tqdm import tqdm
try:
    from patch_pickle import patch_mp_connection_bpo_17560
except:
    from fairseq_cli.patch_pickle import patch_mp_connection_bpo_17560
from multiprocessing import Pool

SEP = 'soDeliveryDate'  # select one unused token as the separator [sep]

SEPbpeidx = 39811

SEPidx = 50222


def _load_preprocess_idx(dataset, division):
    ret = {}
    for div in division:
        with open('%s/%s.preprocess.idx.json' % (dataset, div), 'r', encoding='utf8') as f:
            obj = json.load(f)
        ret[div] = obj
    return ret


def encode_ind_string_lst(line_lst, encode_func, encode_func_params, sep_idx, co_input_truncate=None, output_list=None):
    if co_input_truncate is not None:
        ids_lst = [encode_func(sentence=line[:co_input_truncate], **encode_func_params).tolist()[1:-1] for line in line_lst]
    else:
        ids_lst = [encode_func(sentence=line, **encode_func_params).tolist()[1:-1] for line in line_lst]
    if output_list:
        return ids_lst
    ret_lst = []
    for item_idx, item in enumerate(ids_lst):
        ret_lst.extend(item)
        if item_idx != len(ids_lst) - 1:
            ret_lst.append(sep_idx)
    if ret_lst[-1] != 2:
        ret_lst.append(2)
    ret = torch.LongTensor(ret_lst)
    return ret


class OriginDataset:

    def __init__(self, dataset, division, seed, co_window_size, or_window_size, su_window_size, ma_window_size,
                 co_truncate, or_truncate, fi_truncate, su_truncate, ma_truncate,
                 co_sample_truncate=None, ma_sample_truncate=None,
                 aux_task=None, task=None, model=None, ma_version=None, ma2_exp=None,
                 rouge_version=None, rouge_metric=None, ma2_rm1sub=None,
                 pegasus_task_bool=False,
                 ma_rmcomp=None, ma2_minwin=None, ma2_maxwin=None, ma2_step=None,
                 ma_shuffle=None, ma_mask_p=None, ma_lam=None,
                 co_input_truncate=None,
                 co_input_replace=None,
                 mediasimple=None
                 ):

        def wait_until_done(path):
            import time
            while not os.path.exists(path):
                time.sleep(10)

        patch_mp_connection_bpo_17560()
        self.dataset = dataset
        self.division = division
        self.data = {}
        self.debug = False
        # self.task = task
        # self.model = model
        self.co_window_size = co_window_size
        self.or_window_size = or_window_size
        self.su_window_size = su_window_size
        self.ma_window_size = ma_window_size
        self.co_truncate = co_truncate
        self.or_truncate = or_truncate
        self.fi_truncate = fi_truncate
        self.su_truncate = su_truncate
        self.ma_truncate = ma_truncate
        self.co_sample_truncate = co_sample_truncate
        self.ma_sample_truncate = ma_sample_truncate
        self.ma_version = ma_version
        self.ma2_exp = ma2_exp
        self.rouge_version = rouge_version
        self.rouge_metric = rouge_metric
        self.ma2_rm1sub = ma2_rm1sub
        self.ma_rmcomp = ma_rmcomp
        self.ma2_minwin = ma2_minwin
        self.ma2_maxwin = ma2_maxwin
        self.ma2_step = ma2_step
        self.ma_shuffle = ma_shuffle
        self.ma_mask_p = ma_mask_p  # mask ratio
        self.ma_lam = ma_lam
        self.co_input_truncate = co_input_truncate
        self.co_input_replace = co_input_replace
        self.mediasimple = mediasimple  # reduce memory and speed up for 'mediasum' dataset.
        if self.dataset.startswith('SAMSum'):
            self.preprocessed_idx = _load_preprocess_idx(self.dataset, self.division)
        # self.aux_task = aux_task
        if not self.dataset.startswith('SAMSum'):
            self.data_file = '%s/train.data.aux.pkl' % (self.dataset)
            if os.path.exists(self.data_file):
                print('begin loading datafile... ')
                temp_pkl = self.__load_pkl(self.data_file)
                self.data['train'] = temp_pkl['train']
                self.data['train_origin'] = temp_pkl['train_origin']
                self.data['train_target_origin'] = temp_pkl['train_target_origin']
                assert len(self.data['train']) == len(self.data['train_origin']) and len(self.data['train']) == len(self.data['train_target_origin'])
            else:
                print('begin creating datafile... ')
                self.load(model, self.data_file)
        else:
            self.load()
        print('co_window_size {} {}'.format(co_window_size, '=' * 30))
        print('or_window_size {} {}'.format(or_window_size, '=' * 30))
        print('su_window_size {} {}'.format(su_window_size, '=' * 30))
        print('ma_window_size {} {}'.format(ma_window_size, '=' * 30))
        print('truncated number of sentences {} {}'.format(self.co_truncate, '=' * 30))
        print('truncated number of sentences {} {}'.format(self.or_truncate, '=' * 30))
        print('truncated number of sentences {} {}'.format(self.fi_truncate, '=' * 30))
        print('truncated number of sentences {} {}'.format(self.su_truncate, '=' * 30))
        print('truncated number of sentences {} {}'.format(self.ma_truncate, '=' * 30))
        print('co_sample_truncate {}'.format(co_sample_truncate))
        print('ma2_rm1sub', ma2_rm1sub)
        print('ma_rmcomp', ma_rmcomp)
        print('ma2_maxwin', ma2_maxwin)
        print('ma_version {}'.format(ma_version))
        random.seed(seed)
        np.random.seed(seed)
        if aux_task == 'ma_statistics':
            self.__create_ma_task_v2(task, model)
            return
        if aux_task is not None:
            if 'ma' in aux_task.split('_') and ma_version == 2:
                assert ('su' in aux_task.split('_')) + ('ma' in aux_task.split('_')) == 1
                self.ma_data_file = '%s/train.ma_v2_rouge%s%s%s_mw%s_%s%s%s%saux.pkl' % (self.dataset,
                                                                                         self.rouge_version,
                                                                                         self.rouge_metric,
                                                                                         '_rm1sub' if self.ma2_rm1sub else '',
                                                                                         '%s_%s'%(self.ma2_minwin, self.ma2_maxwin),
                                                                                         ('step%s_'%self.ma2_step) if self.ma2_step is not None else '',
                                                                                         'mashuf_' if ma_shuffle else '',
                                                                                         ('mskp%s_' % self.ma_mask_p) if self.ma_mask_p is not None else '',
                                                                                         'lam%s' % self.ma_lam if self.ma_lam is not None else '')
                assert 'None' not in self.ma_data_file
                if self.debug or self.ma2_exp != '' or not os.path.exists(self.ma_data_file):
                    print('%s not found! begin __create_ma_v2_task... ' % self.ma_data_file)
                    self.__create_ma_task_v2(task, model)
                else:
                    print('begin loading sub_summary... {} {}'.format('=' * 30, self.ma_data_file))
                    self.data['train_sub'] = self.__load_pkl(self.ma_data_file)
                print('after loading/creating, sub_summary sample count {}'.format(sum([len(item) for item in self.data['train_sub'].values()])))
            else:
                assert False
        if not self.dataset.startswith('SAMSum'):
            print('del train_origin')
            del self.data['train_origin']
            print('del train_target_origin')
            del self.data['train_target_origin']
            import gc
            gc.collect()

    def load(self, model=None, data_file=None):
        for div in self.division:
            assert div == 'train'
            if self.dataset.startswith('SAMSum'):
                assert self.dataset in ['SAMSumInd']
                with open('%s/%s.json' % (self.dataset, div), 'r', encoding='utf8') as f:
                    obj = json.load(f)
                self.data[div] = []
                self.data[div+'_target'] = []
                for i in range(len(self.preprocessed_idx[div])):
                    new_idx = self.preprocessed_idx[div][str(i)]
                    utterances = obj[new_idx]['dialogue'].split('\r\n')
                    if len(utterances) <= 1:
                        utterances = obj[new_idx]['dialogue'].split('\n')
                    utterances = [utter.strip() for utter in utterances if len(utter.strip()) != 0]
                    self.data[div+'_target'].append(obj[new_idx]['summary'])
                    # random.shuffle(utterances)
                    # self.data[div].append(' '.join(utterances))
                    self.data[div].append(utterances)
            else:
                # assert self.dataset in ['mediasum']
                with open('%s/%s.bpe.source' % (self.dataset, div), 'r', encoding='utf-8') as f:
                    source_bpe_lines = [line.strip() for line in f.readlines()]
                self.data[div] = []  # [utterance_bpe_id]
                self.data[div + '_origin'] = []  # [utterance_string]
                # [target string]
                with open('%s/%s.target' % (self.dataset, div), 'r', encoding='utf-8') as f:
                    self.data[div + '_target_origin'] = [line.strip() for line in f.readlines()]
                    if self.debug:
                        self.data[div + '_target_origin'] = self.data[div + '_target_origin'][:11]
                    # print('first 1 line of target_origin: ', self.data[div + '_target_origin'][0])
                # [target_bpe_id]
                # with open('%s/%s.bpe.target' % (self.dataset, div), 'r', encoding='utf-8') as f:
                #     self.data[div + '_target'] = [[int(word) for word in line.strip().split()] for line in f.readlines()]
                for dialog_idx, utter_ids in tqdm(enumerate(source_bpe_lines), total=len(source_bpe_lines)):
                    if self.debug and dialog_idx > 10:
                        break
                    utter_items = utter_ids.split(' %s ' % SEPbpeidx)
                    assert len(utter_items) != 0, 'seperator not found! '
                    utterances = []
                    # utterance_ids = []
                    for utter in utter_items:
                        utterances.append(model.my_bpe_decode(utter))
                        # utterance_ids.append([int(word) for word in utter.split()])
                        # print('utterance_ids[-1] = ', utterance_ids[-1])
                    # if dialog_idx == 0:
                    #     print('first dialog utterance_bpe_lst: ', utter_items)
                    #     print('first dialog utterance_lst: ', [utt.encode('utf-8') for utt in utterances])
                    self.data[div].append(utter_items)
                    self.data[div+'_origin'].append(utterances)
                print(len(self.data[div]), len(self.data[div+'_origin']), len(self.data[div+'_target_origin']))
                assert len(self.data[div]) == len(self.data[div+'_origin']) and len(self.data[div]) == len(self.data[div+'_target_origin'])

                with open(data_file, 'wb') as f:
                    pkl.dump({
                        'train': self.data['train'],
                        'train_origin': self.data['train_origin'],
                        'train_target_origin': self.data['train_target_origin']
                    }, f)

    def __load_pkl(self, file):
        with open(file, 'rb') as f:
            return pkl.load(f)

    @staticmethod
    def __create_ma_task_v2_one(self, task, model, rouge, param_lst):
        if task is not None:
            dialog_idx, utterance_lst = param_lst
        else:
            dialog_idx = param_lst
            if rouge is None:
                rouge = self.rouge
            utterance_lst = self.data['train'][dialog_idx]

        if self.dataset.startswith('SAMSum'):
            utterance_lst = [utter.strip() for utter in utterance_lst if len(utter.strip()) != 0]
            summary = self.data['train_target'][dialog_idx]
        else:
            utterance_lst = []
            utterance_bpe_lst = []
            for utter, utter_bpe in zip(self.data['train_origin'][dialog_idx], self.data['train'][dialog_idx]):
                if len(utter.strip()) > 0:
                    utterance_lst.append(utter.strip())
                    utterance_bpe_lst.append(utter_bpe)
            summary = self.data['train_target_origin'][dialog_idx]
            if len(utterance_lst) <= 1:
                return []
        # print('create_ma2, utterance_lst = ', [item.encode('utf-8') for item in utterance_lst])
        # print('create_ma2, summary = ', summary)
        window_high = self.window_high_dict[len(utterance_lst)]
        summ_lst = [summ.strip() for summ in summary.split('.') if len(summ.strip()) > 0]  # rebase [modification 1]
        if ('allsub' not in self.ma2_exp) and len(summ_lst) < 1:  ############################### at least two sub-summaries ##############################################
            return []
        elif ('allsub' not in self.ma2_exp) and self.ma2_rm1sub and len(summ_lst) == 1:
            return []
        ret = []
        for summ_idx, summ in enumerate(summ_lst):
            summ += '.'
            best_size, best_score = None, 0
            best_start, best_end = None, None
            for window_size in range(self.ma2_minwin, window_high + 1):
                if self.ma2_step is None:
                    step = window_size // 2 + (window_size % 2 == 1)
                else:
                    step = self.ma2_step
                if len(utterance_lst) < step + window_size:
                    input('error')
                score_lst = []
                for sub_idx, start in enumerate(
                        range(0, len(utterance_lst), step)):
                    end = min(start + window_size, len(utterance_lst))
                    utter = ' '.join(utterance_lst[start: end])
                    if not self.dataset.startswith('SAMSum') and len(''.join(utter.split('.')).strip()) == 0:
                        continue
                    score = rouge.get_scores(utter, summ)[0]['rouge-%s' % self.rouge_version][self.rouge_metric]
                    score_lst.append((score, window_size, start, end))
                # if self.debug:
                #     print('score_lst', score_lst)
                assert len(score_lst) > 1
                max_idx = int(np.argmax([item[0] for item in score_lst]))
                if score_lst[max_idx][0] > best_score:
                    best_score = score_lst[max_idx][0]
                    best_size = score_lst[max_idx][1]
                    best_start, best_end = score_lst[max_idx][2], score_lst[max_idx][3]
            if best_score <= 0:
                continue
            # if dialog_idx not in self.data['train_sub']:
            #     self.data['train_sub'][dialog_idx] = []
            if self.dataset.startswith('SAMSum'):
                cur_utter_lst = utterance_lst[best_start: best_end]
            else:
                cur_utter_lst = utterance_bpe_lst[best_start: best_end]
            # print('best_s end, ', best_start, best_end)
            # print('bpe: cur_utter_lst', cur_utter_lst)
            if self.ma_shuffle:
                random.shuffle(cur_utter_lst)

            if self.dataset.startswith('SAMSumInd'):
                source_ids = encode_ind_string_lst(cur_utter_lst, model.my_encode, {'task': task}, sep_idx=SEPidx)
                ret.append({
                    'id': int('%s0%s' % (dialog_idx, len(ret))),
                    'source': source_ids,
                    'target': model.my_encode(task, summ)[1:],
                    'border': best_start,
                    'win': best_size
                })
            else:  # self.dataset.startswith('mediasum'):
                if task is None and model is None:
                    ret.append({
                        'id': int('%s0%s' % (dialog_idx, len(ret))),
                        'source': cur_utter_lst,
                        'target': summ,
                        'border': best_start,
                        'win': best_size
                    })
                else:
                    source_ids = encode_ind_string_lst(cur_utter_lst, model.my_encode, {'task': task, 'no_bpe': True}, sep_idx=SEPidx)
                    ret.append({
                        'id': int('%s0%s' % (dialog_idx, len(ret))),
                        'source': source_ids,
                        'target': model.my_encode(task, summ)[1:],
                        'border': best_start,
                        'win': best_size
                    })
        return ret

    @staticmethod
    def create_ma_task_v2_map(one_param):
        return self_obj.__create_ma_task_v2_one(self_obj, None, None, None, one_param)

    def pool_init(self):
        global self_obj
        self_obj = self

    def __create_ma_task_v2(self, task, model):
        print('self.rouge_version = ', self.rouge_version, '#rouge-%s#' % self.rouge_version)
        rouge = Rouge(metrics=['rouge-%s' % self.rouge_version])
        self.data['train_sub'] = {}
        size_dist = {}

        option_window_high_dict = {}
        for dia_len in range(2, 32):
            cur_max = 1
            while True:
                if self.ma2_step is None:
                    step = cur_max // 2 + (cur_max % 2 == 1)
                else:
                    step = self.ma2_step
                if not (dia_len >= cur_max + step):
                    break
                else:
                    cur_max += 1
            cur_max -= 1
            option_window_high_dict[dia_len] = cur_max

        self.window_high_dict = {}
        temp_idx = 2
        while temp_idx <= len(option_window_high_dict) + 1 and option_window_high_dict[temp_idx] <= self.ma2_maxwin:
            self.window_high_dict[temp_idx] = option_window_high_dict[temp_idx]
            temp_idx += 1
        for i in range(temp_idx, 56 if self.dataset.startswith('SAMSum') else 110):  # maximum number of utterances in one dialogue, SAMSum: 56 utters, MediaSum: 109 utters.
            self.window_high_dict[i] = self.ma2_maxwin

        if self.dataset.startswith('SAMSum'):
            print('always single process! ')
            for dialog_idx, utterance_lst in tqdm(enumerate(self.data['train']), total=len(self.data['train'])):  # < 76020, continue
                ret = self.__create_ma_task_v2_one(self, task, model, rouge, [dialog_idx, utterance_lst])
                if len(ret) != 0:
                    self.data['train_sub'][dialog_idx] = ret
        else:  # mediasum dataset
            # assert False, 'can not create sub-summary! '
            self.rouge = rouge
            with Pool(processes=48, initializer=self.pool_init) as pool:
                # param_lst = [[None, None, rouge, [dialog_idx, utterance_lst]] for dialog_idx, utterance_lst in enumerate(self.data['train'][:2])]
                param_lst = list(range(len(self.data['train'])))
                # param_lst = [[self, param] for param in param_lst]
                print('creating param_lst done! ')

                # imap_result = []
                # for param_lst in param_lst:
                #     ret_one = self.__create_ma_task_v2_one(task, model, rouge, param_lst)
                #     imap_result.append(ret_one)

                # ma2_sample_lst = self.__create_ma_task_v2_list(task, model, rouge, param_lst, 300)
                ma2_sample_lst = pool.imap(self.create_ma_task_v2_map, param_lst, 100)
                print('start collecting... ')
                for dialog_idx, sample_lst in tqdm(enumerate(ma2_sample_lst), total=len(param_lst)):
                    new_sample_lst = []
                    for sample_idx, sample in enumerate(sample_lst):
                        source_ids = encode_ind_string_lst(sample['source'], model.my_encode, {'task': task, 'no_bpe': True},sep_idx=SEPidx)
                        sample['source'] = source_ids
                        sample['target'] = model.my_encode(task, sample['target'])[1:]
                        new_sample_lst.append(sample)
                    if len(new_sample_lst) != 0:
                        self.data['train_sub'][dialog_idx] = new_sample_lst

        print('sub_summary sample count {}'.format(sum([len(item) for item in self.data['train_sub'].values()])))
        if self.ma2_exp == '':
            with open(self.ma_data_file, 'wb') as f:
                pkl.dump(self.data['train_sub'], f)

    def shuffle_sents(self, sents, sep_id, replace_idx=None, replace_lst=None):
        # 1. remove the last "2" id
        # 2. seperate sentences based on "sep_id"
        # 3. shuffle sentences
        # 4. add "2" id to the end.
        sents = sents.tolist()
        if sents[-1] == 2:
            sents = sents[:-1]
        new_sents = []
        cur_sent = []
        sep_id_count = 0
        for item in sents:
            if item != sep_id:
                cur_sent.append(item)
            else:
                sep_id_count += 1
                new_sents.append(cur_sent)
                cur_sent = []
        new_sents.append(cur_sent)
        if len(new_sents) == 2:
            new_sents[0], new_sents[1] = new_sents[1], new_sents[0]
        else:
            random.shuffle(new_sents)
        # print('before replace: ', new_sents)
        assert sep_id_count == len(new_sents) - 1
        ret_sents = []
        for sen_idx, sen in enumerate(new_sents):
            if replace_idx is not None and sen_idx in replace_idx:
                # print('add replace %s, %s' % (sen_idx, replace_lst[replace_idx.index(sen_idx)]))
                ret_sents.extend(replace_lst[replace_idx.index(sen_idx)])
            else:
                ret_sents.extend(sen)
            if sen_idx != len(new_sents) - 1:
                ret_sents.append(sep_id)
        ret_sents.append(2)
        return torch.LongTensor(ret_sents)

    def __getitem__(self, task, model, dialog_id, cur_sample_type, aux_method):
        """
            sample_type: Option['coherence', 'margin sub-summary']
        """
        assert aux_method in ['sample', 'batch'] or cur_sample_type == 'MLM_GSG' and aux_method is None
        try:
            dialog_id = dialog_id.item()
        except:
            pass
        if self.dataset.startswith('SAMSum'):
            utterances = copy.deepcopy(self.data['train'][dialog_id])  # the list of utterance_string
        else:
            utterances = self.data['train'][dialog_id]  # the list of bpe_id_string
        assert cur_sample_type in ['co', 'ma']
        ret = [[], []]
        if cur_sample_type in ['co', 'ma'] and len(utterances) > getattr(self, '{}_truncate'.format(cur_sample_type)):
            return ret
        if cur_sample_type in ['co']:
            window_size = self.or_window_size if cur_sample_type == 'or' else self.co_window_size
            option_range = list(range(0, len(utterances), window_size // 2 + (window_size % 2 == 1)))
            if aux_method == 'sample':
                option_range = [np.random.choice(option_range)]
            if self.co_sample_truncate != -1:
                random.shuffle(option_range)
                option_range = option_range[:self.co_sample_truncate]
            for sub_idx, start in enumerate(option_range):
                end = min(start + window_size, len(utterances))
                if end - start < 2:
                    continue
                if cur_sample_type == 'co':
                    if self.dataset.startswith('SAMSumInd'):
                        ret[0].append({
                            'id': int('%s0%s' % (dialog_id, sub_idx)),
                            'source': encode_ind_string_lst(utterances[start: end], model.my_encode, {'task': task}, sep_idx=SEPidx)
                        })
                    else:  # self.dataset.startswith('mediasum'):
                        ret[0].append({
                            'id': int('%s0%s' % (dialog_id, sub_idx)),
                            'source': encode_ind_string_lst(utterances[start: end], model.my_encode, {'task': task, 'no_bpe': True}, sep_idx=SEPidx, co_input_truncate=self.co_input_truncate)
                        })
                # print('original co_sample = ', ret[0][-1])

                p1 = random.random()
                cur_len = min(end, len(utterances)) - start
                replace_option_lst = list(range(0, start)) + list(range(end, len(utterances)))
                if not self.co_input_replace or (self.co_input_replace and (p1 > 0.5 or p1 < 0.5 and len(replace_option_lst) == 0)):
                    sub_neg_utterance_encoded = self.shuffle_sents(ret[0][-1]['source'], SEPidx)
                else:
                    # print('dialog_idx', dialog_id)
                    replace_option_idx = np.random.choice(replace_option_lst, min(int(cur_len * 0.5), len(replace_option_lst)), replace=False)
                    # print('replace_option_lst = ', replace_option_lst)
                    # print('replace_option_idx = ', replace_option_idx)
                    encoded_lst = encode_ind_string_lst([utterances[t_idx] for t_idx in replace_option_idx], model.my_encode, {'task': task, 'no_bpe': True}, sep_idx=SEPidx, output_list=True)
                    # print('replace_lst = ', encoded_lst)
                    replace_idx = np.random.choice(range(cur_len), len(encoded_lst), replace=False).tolist()
                    # print('replace_idx = ', replace_idx)
                    sub_neg_utterance_encoded = self.shuffle_sents(ret[0][-1]['source'], SEPidx, replace_idx=replace_idx, replace_lst=encoded_lst)
                    # print('sub_neg_utterance_encoded', sub_neg_utterance_encoded)
                    # input('wait')


                if self.dataset.startswith('SAMSumInd'):
                    ret[1].append({
                        'id': int('%s0%s' % (dialog_id, sub_idx)),
                        'source': sub_neg_utterance_encoded
                    })
                else:  # self.dataset.startswith('mediasum'):
                    ret[1].append({
                        'id': int('%s0%s' % (dialog_id, sub_idx)),
                        'source': sub_neg_utterance_encoded
                    })

            return ret
        elif cur_sample_type == 'ma':
            pos_lst = self.data['train_sub'].get(dialog_id, [])
            # print(dialog_id, 'pos_lst = ', pos_lst)
            if self.ma_sample_truncate != -1:
                random.shuffle(pos_lst)
                pos_lst = pos_lst[: self.ma_sample_truncate]
            if aux_method == 'sample':
                pos_lst = [np.random.choice(pos_lst)]
            if len(pos_lst) != 0:
                ret[0] = pos_lst
                for sample_idx, sample in enumerate(pos_lst):
                    if self.ma_version == 2 and 'mutual' in self.ma2_exp and len(pos_lst) > 1:
                        option_sample_idx = list(range(0, len(pos_lst)))
                        option_sample_idx.remove(sample_idx)
                        neg_idx = np.random.choice(option_sample_idx)
                        neg_start = pos_lst[neg_idx]['border']
                        neg_end = neg_start + pos_lst[neg_idx]['border']  # neg_end = neg_start + pos_lst[neg_idx]['win'] bug===
                        if 'add1' in self.ma2_exp:
                            neg_start = max(0, neg_start - 1)
                            neg_end = min(len(utterances), neg_end + 1)
                    else:
                        best_start = sample['border']
                        if self.ma_version == 1:
                            option_start = list(range(0, len(utterances), self.ma_window_size // 2 + (self.ma_window_size % 2 == 1)))
                        elif self.ma_version == 2:
                            if self.ma2_step is None:
                                step = sample['win'] // 2 + (sample['win'] % 2 == 1)
                            else:
                                step = self.ma2_step
                            option_start = list(range(0, len(utterances), step))
                        else:
                            raise Exception
                        option_start.remove(best_start)
                        neg_start = np.random.choice(option_start)
                        if self.ma_version == 1:
                            neg_end = neg_start + self.ma_window_size
                        elif self.ma_version == 2:
                            neg_end = neg_start + sample['win']
                            if 'add1' in self.ma2_exp:
                                neg_start = max(0, neg_start - 1)
                                neg_end = min(len(utterances), neg_end + 1)
                        else:
                            raise Exception
                    # negative samples
                    cur_utter_lst = utterances[neg_start: neg_end]
                    if self.ma_shuffle:
                        random.shuffle(cur_utter_lst)
                    if self.dataset.startswith('SAMSumInd'):
                        source_ids = encode_ind_string_lst(cur_utter_lst, model.my_encode, {'task': task}, sep_idx=SEPidx)
                        ret[1].append({
                            'id': sample['id']*10 + 9,
                            'source': source_ids,
                            'target': sample['target']
                        })
                    else:  # self.dataset.startswith('mediasum'):
                        source_ids = encode_ind_string_lst(cur_utter_lst, model.my_encode, {'task': task, 'no_bpe': True}, sep_idx=SEPidx)
                        ret[1].append({
                            'id': sample['id']*10 + 9,
                            'source': source_ids,
                            'target': sample['target']
                        })
            return ret
        else:
            assert False, 'unknown auxiliary task! '

class AverageMeter:

    def __init__(self, name_lst, version):
        self.name_lst = name_lst
        self.metric_lst = {}
        self.version = version
        for name in self.name_lst:
            self.metric_lst[name] = [0, 0, 0, 0]

    def zero_cur(self):
        for name in self.name_lst:
            for i in range(2):
                self.metric_lst[name][i] = 0

    def update_metric(self, log):
        if log is not None:
            if self.version == 1:
                key_lst = list(log.keys())
                key_lst.remove('aux_task')
                for key in key_lst:
                    self.metric_lst[key][0] += log[key]
                    self.metric_lst[key][1] += 1
                    self.metric_lst[key][2] += log[key]
                    self.metric_lst[key][3] += 1
            else:
                key_lst = list(log.keys())
                key_lst.remove('aux_task')
                for key in key_lst:
                    self.metric_lst[key][0] += log[key][0]
                    self.metric_lst[key][1] += log[key][1]
                    self.metric_lst[key][2] += log[key][0]
                    self.metric_lst[key][3] += log[key][1]

    def __get_cur_avg(self, name):
        return self.metric_lst[name][0] / self.metric_lst[name][1]

    def __get_total_avg(self, name):
        return self.metric_lst[name][2] / self.metric_lst[name][3]

    def print(self, epoch_done):
        if not epoch_done:
            output_line_lst = ['| co_b {} ma_b {}'.format(self.metric_lst['co_loss'][1], self.metric_lst['ma_loss'][1])]
            for key in self.name_lst:
                if self.metric_lst[key][1] != 0:
                    output_line_lst.append('| {} {:.4f}'.format(key, self.__get_cur_avg(key)))
        else:
            output_line_lst = ['| co_b {} ma_b {}'.format(self.metric_lst['co_loss'][3], self.metric_lst['ma_loss'][3])]
            for key in self.name_lst:
                if self.metric_lst[key][3] != 0:
                    output_line_lst.append('| {} {:.4f}'.format(key, self.__get_total_avg(key)))
        return ' '.join(output_line_lst)

