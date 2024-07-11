import os
import io
import sys
import time
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, root_path: str = None, file_name: str = None):
        self.root_path = root_path
        self.file_name = file_name
        self.logger = self._generate_logger()
        self.tensorboard_logger = self._build_writer()

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def divider(self, msg, flag='=', end=False, *args, **kwargs):
        self.logger.divider(msg, flag, end, *args, **kwargs)

    def warring(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def close(self):
        self.logger.disabled = True
        self.logger = None

    @staticmethod
    def _divider(self, msg, flag='=', end=False, *args, **kwargs):
        msg_len = len(msg)
        left_msg = flag * int((80 - msg_len) / 2 - 2)
        right_msg = flag * int((80 - msg_len) / 2 - 2)
        msg = left_msg + '> ' + msg + ' <' + right_msg
        if end:
            msg += '\n'
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def _build_writer(self) -> SummaryWriter:
        """
        build writer
        :return:
        """

        if self.root_path is None:
            self.root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')

        writer = SummaryWriter(log_dir=self.root_path + '/runs')

        return writer

    def _generate_logger(self) -> logging.Logger:
        """
        generate logger
        :return: logger
        """

        if self.root_path is None:
            self.root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')

        if self.file_name is None:
            self.file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        log_file_save_path = os.path.join(self.root_path, self.file_name + '.log')

        if not os.path.exists(os.path.dirname(log_file_save_path)):
            os.makedirs(os.path.dirname(log_file_save_path))

        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s')

        # set file handler
        file_handler = logging.FileHandler(log_file_save_path, encoding='utf-8')
        file_handler.setFormatter(formatter)

        # set stream handler
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        # generate logger
        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program)
        logger.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logging.Logger.divider = self._divider

        return logger
