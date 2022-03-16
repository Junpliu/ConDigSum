import argparse
from tqdm import tqdm as tqdm
import os


def test(log_dir, dataset, beam, min_len, lenpen, no_repeat_ngram_size, bsz, max_len=None):
    import torch
    from fairseq.models.bart import BARTModel

    bart = BARTModel.from_pretrained(
        log_dir,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=dataset
    )

    bart = bart.cuda()
    bart.eval()
    hyp_path = '%s/test.hypo' % (log_dir)
    with open('%s/test.bpe.source' % (dataset), 'r', encoding='utf-8') as f:
        dialog_lst = [line.strip() for line in f.readlines()]
    with torch.no_grad():
        with open(hyp_path, 'w', encoding='utf-8') as fout:
            for i in tqdm(range(0, len(dialog_lst), bsz), total=len(range(0, len(dialog_lst), bsz))):
                slines = dialog_lst[i: i+bsz]
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=max_len,
                                               min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--bsz', default=16)
    args = parser.parse_args()

    args = vars(args)
    if args['dataset'].startswith('SAMSum'):
        args['max_len'] = 100
        args['beam'] = 4
        args['min_len'] = 3
        args['lenpen'] = 0.5
        args['no_repeat_ngram_size'] = 3
    else:
        args['max_len'] = 80
        args['beam'] = 3
        args['min_len'] = 3
        args['lenpen'] = 0.1
        args['no_repeat_ngram_size'] = 2

    os.system('mkdir -p %s/output' % args['log_dir'])
    test(**args)
