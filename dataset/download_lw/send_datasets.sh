machine=Lightwheel_A40
dataset=1W_Libero_X7s
dataset_dir=/home/erdao/Documents/LightwheelData/$dataset/
remote_dir=/home/lightwheel/erdao.liang/$dataset/

# first time: ssh-copy-id $machine

rsync -avz --progress --partial --inplace --timeout=60 \
    $dataset_dir \
    $machine:$remote_dir