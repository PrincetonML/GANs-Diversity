# GANs-Diversity
Code for Birthday Paradox test for diversity of GANs, proposed in [Do GANs actually learn the distribution? An empirical study](https://arxiv.org/abs/1706.08224)

based on [DCGAN-Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

## Requirement
- Python, Numpy, Tensorflow

## Get collision candidates of generated faces from a DCGAN

Download the model checkpoint and unzip it

	$ wget https://www.dropbox.com/s/sfdoyvfl5eozfa5/checkpoint.zip
	$ unzip checkpoint.zip

Generate a pool of $topK=20$ candidate collision pairs from a batch of 6*64=384

	$ python main.py --topK=20 --num_batches=6 --sample_dir=samples

The top 20 potential collisions will be in ./samples

## Example code for collision candiate selection

```python
import numpy as np
import heapq as hq
import copy as cp



images = sample_from_gan()	# a batch of generated samples
queue = []	# a priority queue maintaining top K most similar pairs
n_image = images.shape[0]
topK = 20 # keep top

for i in range(n_image):
for j in range(i+1, n_image):
  # measure similarity in pixel space (could be done in some embedding space too)
  dist = np.sum((images[i] - images[j])**2)
  if len(queue) == 0 or -1*dist > queue[0][0]:
    hq.heappush(queue, (-1*cp.deepcopy(dist), cp.deepcopy(images[i]), cp.deepcopy(images[j])))
    if len(queue) > topK:
      hq.heappop(queue)

for idx in range(topK):
	neg_dist, img1, img2 = hq.heappop(queue)
	scipy.misc.imsave(config.sample_dir + '/pair#%d_%f_%d.png'%(idx, -1*neg_dist, 1), (img1+1.)/2)
	scipy.misc.imsave(config.sample_dir + '/pair#%d_%f_%d.png'%(idx, -1*neg_dist, 2), (img2+1.)/2)
```