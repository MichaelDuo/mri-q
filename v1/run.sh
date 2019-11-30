# make
# ./mri-q -i ../datasets/large/input/64_64_64_dataset.bin -o output.bin
# python ../tools/compare-output output.bin ../datasets/large/output/64_64_64_dataset.out

make
./mri-q -i ../datasets/128x128x128/input/128x128x128.bin -o output.bin
python ../tools/compare-output output.bin ../datasets/128x128x128/output/blah