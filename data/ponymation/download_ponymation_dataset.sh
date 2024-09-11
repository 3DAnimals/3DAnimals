echo "----------------------- downloading horse dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/ponymation/horse.zip
echo "----------------------- unzipping horse dataset -----------------------"
unzip -q horse.zip

echo "----------------------- downloading cow dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/ponymation/cow.zip
echo "----------------------- unzipping cow dataset -----------------------"
unzip -q cow.zip

echo "----------------------- downloading zebra dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/ponymation/zebra.zip
echo "----------------------- unzipping horse dataset -----------------------"
unzip -q zebra.zip

echo "----------------------- downloading giraffe dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/ponymation/giraffe.zip
echo "----------------------- unzipping giraffe dataset -----------------------"
unzip -q giraffe.zip

echo "----------------------- downloading horse stage2 dataset -----------------------"
wget https://download.cs.stanford.edu/viscam/3DAnimals/data/ponymation/horse_stage2.tgz
echo "----------------------- unzipping horse stage 2 dataset -----------------------"
tar -xvf horse_stage2.tgz