rm -rf *.npy
rm -rf model_res_sample.txt
rm -rf nohup.out
rm -rf img/*
rm -rf .save/*
rm -rf debug/att/*
rm -rf debug/enc/*
rm -rf debug/hid/*
rm -rf debug/mask/*
nohup python train.py 2>&1 &
