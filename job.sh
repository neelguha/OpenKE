#!/bin/bash

python grid_search.py -d uniform500_wd -f ./../mb_json/datasets/uniform500/splits/OpenKE_wd/ -p TransE -s 50
python grid_search.py -d uniform500_mbz -f ./../mb_json/datasets/uniform500/splits/OpenKE_mbz/ -p TransE -s 50

python grid_search.py -d popular500_wd -f ./../mb_json/datasets/popular500/splits/OpenKE_wd/ -p TransE -s 50
python grid_search.py -d popular500_mbz -f ./../mb_json/datasets/popular500/splits/OpenKE_mbz/ -p TransE -s 50
