from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Test Option Class"""

    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--load_checkpoint_path', required=True, type=str, help='checkpoint path')
        self.parser.add_argument('--save_result_path', default=None, type=str, help='save result path')
        self.parser.add_argument('--max_val_samples', default=None, type=int, help='max val data')
        self.parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
        self.parser.add_argument('--exec_guess_ans', default=1, type=int, help='If True, the ClevrExecutor guesses '
                                                                                    'answer from answer candidates')
        self.parser.add_argument('--is_baseline_model', default=0, type=int, help='True if test model being loaded'
                                                                                  'is a baseline model (ns-vqa')
        self.is_train = False
