Fun code is in the source folder.

PLEASE NOTE THAT THIS IS A WIP THAT I STOPPED WORKING ON ABOUT A YEAR AGO SO THE CURRENT STATE OF THE PROJECT IS UNKNOWN. PLEASE CONTINUE DEVELOPMENT IF YOU SO CHOOSE AND LET ME KNOW WHAT CHANGES YOU MAKE; I'M CURIOUS IF THIS HAS ANY PRACTICAL APPLICATION. IT WAS FUN TO WORK ON WHILE I DID AND MAYBE ONE DAY I WILL FINALIZE THE TURN AND START WORK ON THE FLOP CLUSTERS TO BE APPLIED TO SOME MACHINE LEARNING APPLICATION...

THE POKER.YML FILE IS FOR ANACONDA AND LISTS ALL THE DEPENDENCIES I'VE USED FOR MY EXPLORATIONS INTO POKER.

WHAT'S INCLUDED IN THE YML FILE IS OVERKILL FOR THIS PROJECT, BUT I'M TOO LAZY TO FILTER OUT WHAT ISN'T NECESSARY... AT THE END OF THE DAY, THE PROJECT WILL COMPILE IF YOU USE THAT YML FILE LOL

If you modify ccluster.pyx or kmeans.pyx, be sure to rebuild using:
```
python ccluster-helper.py build_ext --inplace
```

To see what uses python, execute the below command and open ccluster.html in the source directory. 
```
cython ccluster.pyx -a
```
Python dense areas will he highlighted yellow.


Generates a (semi-sparce) 67+ gb numpy matrix of the following format:
(hcrd = hand card; tcrd = table card)

```
hcrd1|hcrd2|tcrd1|tcrd2|tcrd3|tcrd4|tcrd5|tie #|win #|loss#
int16|int16|int16|int16|int16|int16|int16|int16|int16|int16
```

Elements of the above matrix can be accessed using two mappings. One maps your hand to a particular chunk. The other maps the board to an index within the chunk. This scales and will be more intuitive when you see it. Point being that accessing an element most nearly O(1)...

Essentially, (this may not be completely accurate, but conceptually equivalent) the flow of data is as follows:

// comments are not part of dataset...

```
hcrd1|hcrd2|tcrd1|tcrd2|tcrd3|tcrd4|tcrd5|tie #|win #|loss#
0    |0    |0    |0    |0    |0    |0    |###  |###  |### // BEGINNING OF 'CHUNK' 0
0    |0    |1    |0    |0    |0    |0    |###  |###  |###
0    |0    |2    |0    |0    |0    |0    |###  |###  |###
... ...
0    |0    |51   |51   |51   |51   |50   |###  |###  |###
0    |0    |51   |51   |51   |51   |51   |###  |###  |###   // END OF 'CHUNK' 0 
1    |0    |0    |0    |0    |0    |0    |###  |###  |###   // BEGINNING OF 'CHUNK' 1
1    |0    |1    |0    |0    |0    |0    |###  |###  |###
1    |0    |2    |0    |0    |0    |0    |###  |###  |###
... ... ...
```

using this memory representation of hands you can construct lables for the cluster a particular hand falls under and access it remarkably fast (if you have enough RAM to hold the entire dataset without using SWAP).

This lets you access all hand strength calculations in a pruned dataset (named 'fluffy' that is about 24 gigs) for use in turn/flop calculations.

Current stats project an ~3 hour runtime to generate exact hand strength calculations for the river on an 8 core system. To deal with the memory demands you'll need to make a 100G + size swap file located in '/var/swapfile' on an SSD with decent speeds (as long as you're not writing from 64 workers...). All datapoints will be aggregated into one file on the fly and access will be done through numpy memory mappings. Increasing the concurrent futures count beyond the number of threads on your system you will reduce your overall runtime because of cpu work loads being stacked.

On my desktop (AMD Ryzon 7 3700x, 64gb ram) with the work split across 15 threads (1 thread reserved for system functions) the total run time is cut down to just under 40 minutes. The process can be fully completed locally with no need for AWS or Google VMs.

Clustering is anticipated to be a two step process. With the first step being to generate a sublinear centroid array to serve as a seed to a primary clustering. Both will be kmean clustering variations.

Similar logic can be applied to the turn and flop datasets while referencing the river dataset for twl (tie win loss) stats. Parallelism will need to be done using prange in cython as we will only be able to store one river dataset in memory at a time.