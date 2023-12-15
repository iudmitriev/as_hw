echo "Downloading dataset..."
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
echo "Finished download, staring unpacking"
mkdir -p data/datasets
unzip LA.zip -d  data/datasets/ >> /dev/null
mv data/datasets/LA data/datasets/asv_dataset 
rm LA.zip

echo "Finished unpacking, staring to reorder files"
python3 scripts/process_dataset.py

rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_asv_protocols
rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_asv_scores
rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_cm_protocols
rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_dev
rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_eval
rm -rf data/datasets/asv_dataset/ASVspoof2019_LA_train
rm -rf data/datasets/asv_dataset/README.LA.txt

echo "Finished downloading dataset!"
