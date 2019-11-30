make
./mri-q-cuda -i ../datasets/large/input/64_64_64_dataset.bin -o output.bin
python ../tools/compare-output output.bin ../datasets/large/output/64_64_64_dataset.out