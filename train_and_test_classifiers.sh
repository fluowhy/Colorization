#!/usr/bin/env bash
python main_clf.py --d cuda:0 --e 20 --bs 100 --lr 2e-4 --ds original
python main_clf.py --d cuda:0 --e 20 --bs 100 --lr 2e-4 --ds mine
python main_clf.py --d cuda:0 --e 20 --bs 100 --lr 2e-4 --ds other

python test_clf.py --d cuda:0 --ds original
python test_clf.py --d cuda:0 --ds mine
python test_clf.py --d cuda:0 --ds other