echo "----------------------- downloading horse video dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/horse_videos_multi.zip
echo  "----------------------- unzipping horse video dataset -----------------------"
unzip -q horse_videos_multi.zip

echo "----------------------- downloading horse combined (video+image) dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/horse_combined.zip
echo "----------------------- unzipping horse combined (video+image) dataset -----------------------"
unzip -q horse_combined.zip

echo "----------------------- downloading COCO cow dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/cow_coco.zip
echo "----------------------- unzipping COCO cow dataset -----------------------"
unzip -q cow_coco.zip

echo "----------------------- downloading COCO giraffe dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/giraffe_coco.zip
echo "----------------------- unzipping COCO giraffe dataset -----------------------"
unzip -q giraffe_coco.zip

echo "----------------------- downloading COCO zebra dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/zebra_coco.zip
echo "----------------------- unzipping COCO zebra dataset -----------------------"
unzip -q zebra_coco.zip

echo "----------------------- downloading bird video dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/magicpony/bird_videos_bonanza.zip
echo "----------------------- unzipping bird video dataset -----------------------"
unzip -q bird_videos_bonanza.zip