VM setup:

n2-highcpu-16 16 vCPU, 8 core, 16 GB memory

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install networkx numpy pandas matplotlib tqdm


run file:

python3 main.py \
  --uva-csv-path uva_dare_year_authors.csv \
  --tau-values 0.05,0.10,0.15,0.20,0.25,0.30 \
  --seed-fractions 0.0025,0.005,0.01,0.02,0.05 \
  --n-runs 10 \
  --max-steps 200 \
  --n-workers 14 \
  --betweenness-k 200 \
  --output-prefix ass2 \
  --save-representative-adoption-times


  python3 main.py \
  --uva-csv-path uva_dare_year_authors.csv \
  --tau-values 0.05,0.10,0.15,0.20,0.25,0.30 \
  --seed-fractions 0.0025,0.005,0.01,0.02,0.05 \
  --n-runs 10 \
  --max-steps 200 \
  --n-workers 14 \
  --betweenness-k 200 \
  --output-prefix ass2_uniform \
  --threshold-mode uniform

runvis:

python3 plots.py --prefix ass2
python plots.py --prefix ass2


