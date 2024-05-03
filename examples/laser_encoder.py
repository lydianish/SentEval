# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Clone RoLASER repo here: https://github.com/lydianish/rolaser.git
And follow instructions for loading the model used in batcher
"""

from __future__ import absolute_import, division, unicode_literals

import sys, os
import logging

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_FAIRSEQ = f"{os.environ['SCRATCH']}/fairseq"
PATH_TO_LASER = f"{os.environ['SCRATCH']}/LASER/source"
PATH_TO_ROLASER = f"{os.environ['SCRATCH']}/RoLASER"

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# import LaserEncoder class
sys.path.insert(0, PATH_TO_FAIRSEQ)
sys.path.insert(0, PATH_TO_LASER)
sys.path.insert(0, PATH_TO_ROLASER)
from rolaser import RoLaserEncoder

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['rolaser'].encode(batch)
    return embeddings

# Load RoLASER model
laser = RoLaserEncoder(
    model_path=f"{PATH_TO_ROLASER}/LASER/models/laser.pt",
    vocab=f"{PATH_TO_ROLASER}/LASER/models/laser.cvocab",
    tokenizer="spm"
)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['laser'] = laser

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    #transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                  'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                  'OddManOut', 'CoordinationInversion']
    probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(probing_tasks)
    print(results)
