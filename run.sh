#!/bin/bash

DATADIR="data/1850"
RUN=r1
WIDTH=640
HEIGHT=480
PATCHSIZE=256
# python label_ui.py --image-dir $DATADIR/train  --label-db $DATADIR/train.db --width $WIDTH --height $HEIGHT


./materialise_label_db.py \
 --label-db $DATADIR/train.db \
 --directory $DATADIR/labels/ \
 --width $WIDTH --height $HEIGHT --label-rescale 0.25

./materialise_label_db.py \
 --label-db $DATADIR/test.db \
 --directory $DATADIR/labels/ \
 --width $WIDTH --height $HEIGHT --label-rescale 0.25

./train.py \
--run $RUN \
--steps 100000 \
--train-steps 1000 \
--train-image-dir $DATADIR/train/ \
--test-image-dir $DATADIR/test/ \
--label-dir $DATADIR/labels/ \
--no-use-batch-norm --no-use-skip-connections \
--width $WIDTH --height $HEIGHT \
--patch-width-height $PATCHSIZE --label-rescale 0.25

./generate_graph_pbtxt.py \
 --no-use-skip-connections --no-use-batch-norm \
 --width 256 --height 256 \
 --pbtxt-output bnn_graph.predict.pbtxt

 python3 -m tensorflow.python.tools.freeze_graph \
 --clear_devices \
 --input_graph bnn_graph.predict.pbtxt \
 ***FIX***THIS****
 --input_checkpoint ckpts/$RUN/20180920_004634 \
 --output_node_names "train_test_model/d4/BiasAdd" \
 --output_graph graph.frozen.pb