#!/usr/bin/env bash
#
# pull_data.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#

'''
borrow from: https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/pull_data.sh
'''

mkdir -p ./data

wget -c 'https://www.dropbox.com/s/ncfa5t950gvtaeb/test.enc?dl=0' -O ./data/test.query
wget -c 'https://www.dropbox.com/s/48ro4759jaikque/test.dec?dl=0' -O ./data/test.answer
wget -c 'https://www.dropbox.com/s/gu54ngk3xpwite4/train.enc?dl=0' -O ./data/train.query
wget -c 'https://www.dropbox.com/s/g3z2msjziqocndl/train.dec?dl=0' -O ./data/train.answer
