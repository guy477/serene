Fun code is in the source folder.

If you modify ccluster.pyx or kmeans.pyx, be sure to rebuild using:
```
python ccluster-helper.py build_ext --inplace
```

To see what uses python, execute the below command and open ccluster.html in the source directory. 
```
cython ccluster.pyx -a
```
Python dense areas will he highlighted yellow.


Will generate a (semi-sparce) 67+ gb numpy matrix.

Elements can be accessed using two mappings. One maps your hand to a particular chunk. The other maps the board to an index within the chunk. This scales and will be more intuitive when you see it. Point being that element access is not quite O(1)... but most nearly.

Current stats project an ~3 hour runtime to generate exact hand strength calculations for the river on an 8 core system. To deal with the memory demands you'll need to make a 100G + size swap file located in '/var/swapfile' on an SSD with decent speeds (as long as you're not writing from 64 workers...). All datapoints will be aggregated into one file on the fly and access will be done through numpy memory mappings. Increasing the concurrent futures count beyond the number of threads on your system you will reduce runtime your overall runtime because of cpu work loads being stacked.

On my desktop (AMD Ryzon 7 3700x, 16gb ram w/128gb swap) with the work split across 15 threads (1 thread reserved for system functions) the total run time is cut down to just around 40 minutes. The process can be fully completed locally with no need for AWS or Google VMs.

Clustering is anticipated to be a two step process. With the first step being to generate a sublinear centroid array to serve as a seed to a primary clustering. Both will be kmean clustering variations

Similar logic can be applied to the turn and flop datasets while referencing the river dataset for twl (tie win loss) stats. Parallelism will need to be done using prange in cython as we will only be able to store one river dataset in memory at a time.