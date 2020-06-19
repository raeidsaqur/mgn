import os, sys
import json
from typing import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger('imported_module').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from rsmlkit.logging import get_logger, set_default_level
logger = get_logger(__file__)
set_default_level(logging.DEBUG)

_dir = os.getcwd()
if _dir not in sys.path:
    sys.path.insert(0, _dir)
    # sys.path.append(os.path.abspath("."))
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils

import wandb
import clevr_parser

graph_parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                       has_spatial=True,
                                       has_matching=True).get_backend(identifier='spacy')
embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')

def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i + 1]:
            return False
        if pred[i] == 2:
            break
    return True


def log_params(opt, result, t=None):
    if opt.visualize_training_wandb and wandb is not None:
        val_qfn = opt.clevr_val_question_filename
        model_path = opt.load_checkpoint_path
        if val_qfn:
            wandb.log({"Info": wandb.Table(data=[[val_qfn], [model_path]],
                                           columns=["val_question_filename",
                                                    "load_checkpoint_path"])})
        results_table = wandb.Table(columns=list(result.keys()))
        results_table.add_data(*tuple(result.values()))
        wandb.log({"Results": results_table})

        checkpoint = {
            'args': opt.__dict__
        }
        for k, v in result.items():
            wandb.log({k: v}, step=t)
            # wandb.log({k: v, 'global_step': 1})
        # checkpoint.update(result)
        checkpoint['result'] = result
        json_fp = os.path.join(opt.run_dir, f'test_opt_{opt.run_timestamp}.json')
        with open(json_fp, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        wandb.save(json_fp)
        #os.remove(json_fp)

opt = TestOptions().parse()
opt.is_train = False
loader = get_dataloader(opt, 'val')
executor = get_executor(opt, graph_parser=graph_parser, embedder=embedder)
model = Seq2seqParser(opt)

logging.debug('| running test')
stats = {
    'count': 0,
    'count_tot': 0,
    'exist': 0,
    'exist_tot': 0,
    'compare_num': 0,
    'compare_num_tot': 0,
    'compare_attr': 0,
    'compare_attr_tot': 0,
    'query': 0,
    'query_tot': 0,
    'correct_ans': 0,
    'correct_prog': 0,
    'total': 0
}
# sample_stats = {
#     'question': '',
#     'q_len': 0,
#     'q_type': '',
#     'prog_seq': [],
#     'pred_prod_seq': [],
#     'pred_answer': '',
#     'answer': '',
#     'img_idx': 0,
# }
sample_stats = {
    'q_lens': [],
    'q_lens_corr': [],
    'q_lens_incorr': []
}

if opt.visualize_training_wandb:
    # WandB: Log metrics with wandb #
    wandb_proj_name = opt.wandb_proj_name
    wandb_identifier = opt.run_identifier
    wandb_name = f"{wandb_identifier}"
    logger.info(f"Directing result to WandB project: ${wandb_proj_name}")
    wandb.init(project=wandb_proj_name, name=wandb_name, notes="Running from mgn.reason.tools.run_test.py")
    wandb.config.update(opt)
    wandb.watch(model.seq2seq)
    table = wandb.Table(columns=list(stats.keys()))

    sample_table_cols = ["Img Index", "Question", "Q len", "Q type", "True Prog", "Pred. Prog", "Pred. Answer", "Answer"]
    sample_table = wandb.Table(columns=sample_table_cols)
    sample_table_correct = wandb.Table(columns=sample_table_cols)
    sample_table_incorrect = wandb.Table(columns=sample_table_cols)

for x, y, ans, idx, g_data in loader:
    bsz = opt.batch_size
    model.set_input(x, y, g_data)
    pred_program = model.parse()
    x_np, y_np, pg_np, img_idx, ans_np = x.numpy(), y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()

    if g_data is not None:
        batch_s, batch_t = g_data[0], g_data[1]
        g_embd, Phi = model.gnn(batch_s.x, batch_s.edge_index_s, batch_s.edge_attr_s, batch_s.batch,
                  batch_t.x, batch_t.edge_index_s, batch_t.edge_attr_s, batch_t.batch)
        data_s_list = batch_s.to_data_list()
        data_t_list = batch_t.to_data_list()

    # Process each x_i in batch (bsz=256) #
    for i in range(pg_np.shape[0]):
        xi = x_np[i]
        yi = y_np[i]
        ans_i = ans_np[i]
        pred_pg_i = pg_np[i]
        if g_data is not None:
            data_si = data_s_list[i]
            data_ti = data_t_list[i]

        qi = " ".join([executor.vocab['question_idx_to_token'][x] for x in xi if x > 0])
        pred_ans = executor.run(pred_pg_i, img_idx[i], 'val',
                                guess=opt.exec_guess_ans,
                                debug=True, data_s=data_si, data_t=data_ti,
                                debug_q=qi)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_i]
        q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][yi[1]])
        is_correct = pred_ans == gt_ans
        if is_correct:
            stats[q_type] += 1
            stats['correct_ans'] += 1
        if check_program(pred_pg_i, yi):
            stats['correct_prog'] += 1

        stats['%s_tot' % q_type] += 1
        stats['total'] += 1
        # Collect and log sample stats #
        qi = " ".join([executor.vocab['question_idx_to_token'][x] for x in xi if x > 0])
        q_len = len([x for x in xi if x > 0])
        prog_seq = utils.get_prog_from_seq(yi, executor.vocab)
        pred_prog_seq = utils.get_prog_from_seq(pred_pg_i, executor.vocab)

        sample_stats['q_lens'].append(q_len)
        _k = 'q_lens_corr' if is_correct else 'q_lens_incorr'
        sample_stats[_k].append(q_len)

        if opt.visualize_training_wandb:
            sample_table.add_data(img_idx[i], qi, q_len, q_type, prog_seq, pred_prog_seq, pred_ans, gt_ans)
            if is_correct:
                sample_table_correct.add_data(img_idx[i], qi, q_len, q_type, prog_seq, pred_prog_seq, pred_ans, gt_ans)
            else:
                sample_table_incorrect.add_data(img_idx[i], qi, q_len, q_type, prog_seq, pred_prog_seq, pred_ans, gt_ans)

    # Collect and log Batch Stats #
    logging.debug('| %d/%d questions processed, accuracy %f' % (
    stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))
    if opt.visualize_training_wandb:
        table.add_data(*tuple(stats.values()))

if "compare_mat_spa" in opt.clevr_val_question_path:
    result = {
        'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "compare_mat" in opt.clevr_val_question_path:
    result = {
        'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "embed_spa_mat" in opt.clevr_val_question_path:
    result = {
        'exist_acc': stats['exist'] / stats['exist_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "embed_mat_spa" in opt.clevr_val_question_path:
    result = {
        'exist_acc': stats['exist'] / stats['exist_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "and_mat_spa" in opt.clevr_val_question_path:
    result = {
        'query_acc': stats['query'] / stats['query_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "or_mat_spa" in opt.clevr_val_question_path:
    result = {
        'count_acc': stats['count'] / stats['count_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
elif "or_mat" in opt.clevr_val_question_path:
    result = {
        'count_acc': stats['count'] / stats['count_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
else:
    result = {
        'count_acc': stats['count'] / stats['count_tot'],
        'exist_acc': stats['exist'] / stats['exist_tot'],
        'compare_num_acc': stats['compare_num'] / stats['compare_num_tot'],
        'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
        'query_acc': stats['query'] / stats['query_tot'],
        'program_acc': stats['correct_prog'] / stats['total'],
        'overall_acc': stats['correct_ans'] / stats['total']
    }
logging.debug(result)
if opt.visualize_training_wandb:
    # Results (and corresponding options) are available on wandb sever in test_opt_{ts}.json file dump
    val_qfn = opt.clevr_val_question_filename.split(".")[0]
    wandb.log({"Batch Stats": table})
    wandb.log({f"Sample Stats": sample_table})
    wandb.log({f"Sample Stats [{val_qfn}] (Correct)": sample_table_correct})
    wandb.log({f"Sample Stats [{val_qfn}] (Incorrect)": sample_table_incorrect})
    wandb.log({"q_length_corr": wandb.Histogram(sample_stats['q_lens_corr'], num_bins=10 )})
    wandb.log({"q_length_incorr": wandb.Histogram(sample_stats['q_lens_incorr'], num_bins=10)})
    log_params(opt, result)
else:
    # Record local copies of results #
    utils.mkdirs(os.path.dirname(opt.save_result_path))
    with open(opt.save_result_path, 'w') as fout:
        json.dump(result, fout, indent=2)
    logging.debug('| result saved to %s' % opt.save_result_path)
