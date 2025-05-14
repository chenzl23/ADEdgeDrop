python ./main.py --dataset-name pubmed --backbone GCN --thred 0.7 --alpha 0.2 --gamma 0.3 --feature-normalize 1
python ./main.py --dataset-name ACM --backbone GCN --thred 0.8 --alpha 0.4 --gamma 0.5 --feature-normalize 1
python ./main.py --dataset-name Chameleon --backbone GCN --thred 1 --alpha 0.3 --gamma 0.6 --feature-normalize 1
python ./main.py --dataset-name UAI --backbone GCN --thred 0.9 --alpha 1 --gamma 0.7 --feature-normalize 1
python ./main.py --dataset-name Actor --backbone GCN --feature-normalize 0 --thred 0.9 --alpha 0.1 --gamma 0.6
python ./main.py --dataset-name BlogCatalog --backbone GCN --thred 0.8 --alpha 0 --gamma 0.9 --feature-normalize 0
python ./main.py --dataset-name arxiv --backbone GCN --thred 0.9 --alpha 0.7 --gamma 0.5 --hidden-dim 256 --feature-normalize 1 --epoch-num 500 --drop-line-graph 0.95
python ./main.py --dataset-name CoraFull --backbone GCN --thred 0.9 --alpha 0.6 --gamma 0.1 --feature-normalize 1

python ./main.py --dataset-name pubmed --backbone GAT --thred 0.8 --alpha 0.6 --gamma 0.7 --feature-normalize 1
python ./main.py --dataset-name ACM --backbone GAT --thred 0.8 --alpha 0.9 --gamma 0.8 --feature-normalize 1
python ./main.py --dataset-name Chameleon --backbone GAT --thred 0.7 --alpha 0.6 --gamma 0.4 --feature-normalize 1
python ./main.py --dataset-name UAI --backbone GAT --thred 1 --alpha 1 --gamma 0.8 --feature-normalize 1
python ./main.py --dataset-name Actor --backbone GAT --thred 0.9 --alpha 0 --gamma 0.5 --feature-normalize 0
python ./main.py --dataset-name BlogCatalog --backbone GAT --thred 0.8 --alpha 0.1 --gamma 0.9 --feature-normalize 0
python ./main.py --dataset-name arxiv --backbone GAT --thred 0.8 --alpha 0.9 --gamma 0.5 --hidden-dim 256 --feature-normalize 1 --epoch-num 500 --drop-line-graph 0.95
python ./main.py --dataset-name CoraFull --backbone GAT --thred 0.9 --alpha 0.8 --gamma 0.6 --feature-normalize 1

python ./main.py --dataset-name pubmed --backbone SAGE --thred 0.7 --alpha 0.2 --gamma 0.3 --feature-normalize 1
python ./main.py --dataset-name ACM --backbone SAGE --thred 0.7 --alpha 0.2 --gamma 0.4 --feature-normalize 1
python ./main.py --dataset-name Chameleon --backbone SAGE --thred 0.7 --alpha 0.4 --gamma 0.8 --feature-normalize 0
python ./main.py --dataset-name Actor --backbone SAGE --thred 0.9 --alpha 0.7 --gamma 0.1 --feature-normalize 0
python ./main.py --dataset-name UAI --backbone SAGE --thred 0.7 --alpha 0.2 --gamma 1 --feature-normalize 1
python ./main.py --dataset-name BlogCatalog --backbone SAGE --thred 0.8 --alpha 0.2 --gamma 0.8 --feature-normalize 0
python ./main.py --dataset-name arxiv --backbone SAGE --thred 0.8 --alpha 0.7 --gamma 0.9 --hidden-dim 256 --feature-normalize 1 --epoch-num 500 --drop-line-graph 0.95
python ./main.py --dataset-name CoraFull --backbone SAGE --thred 0.7 --alpha 0.2 --gamma 0.2 --feature-normalize 1