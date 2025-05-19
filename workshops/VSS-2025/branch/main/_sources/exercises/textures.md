---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: plenoptic_venv
  display_name: plenoptic_venv
  language: python
---

# Texture synthesis

```{code-cell} ipython3
import plenoptic as po
import torch
# needed for the plotting/animating:
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['animation.html'] = 'html5'
# use single-threaded ffmpeg for animation writer
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The texture model requires a slightly different setup than the other models we were looking at. In particular, we use coarse-to-fine synthesis, as originally described in the Portilla and Simoncelli paper, which starts with coarsest scales (i.e., lowest spatial frequencies) of the model representation and moves to finer and finer scales.

In plenoptic, this is handled with a different object, [`MetamerCTF`](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.synthesize.html#plenoptic.synthesize.metamer.MetamerCTF). This object is interacted with in the same manner as `Metamer`, except it has several additional arguments for `synthesize` which control how we move between scales. The setup below indicates that we should optimize each scale for 3 iterations and then move on.

```{code-cell} ipython3
img = po.data.reptile_skin().to(DEVICE)
ps = po.simul.PortillaSimoncelli(img.shape[-2:])
ps.to(DEVICE)
im_init = torch.rand_like(img) * .2 + img.mean()
met = po.synth.MetamerCTF(img, ps, loss_function=po.tools.optim.l2_norm)
met.setup(im_init)
met.synthesize(max_iter=500, store_progress=10,
               change_scale_criterion=None,
               ctf_iters_to_check=3)
```

```{code-cell} ipython3
po.synth.metamer.plot_synthesis_status(met);
```

And let's view that synthesis over time:

```{code-cell} ipython3
met.to('cpu')
po.synth.metamer.animate(met)
```

## Different target image

As we practiced earlier, we can change the target image for metamer synthesis straightforwardly. What does it look like to use a different texture? A non-texture image? Are any of these results surprising?

:::{admonition} More texture images
:class: hint

If you run the following lines, you can download some additional texture images used in the original Portilla and Simoncelli paper for use with the model:

If the following code gives you an error, make sure that `pooch` is installed in your virtual environment.
:::

```{code-cell} ipython3
from plenoptic.data.fetch import fetch_data
texture_path = fetch_data("portilla_simoncelli_images.tar.gz")

natural = [
    "3a",
    "6a",
    "8a",
    "14b",
    "15c",
    "15d",
    "15e",
    "15f",
    "16c",
    "16b",
    "16a",
]
natural = po.load_images([texture_path / f"fig{num}.jpg" for num in natural])
artificial = ["4a", "4b", "14a", "16e", "14e", "14c", "5a"]
artificial = po.load_images([texture_path / f"fig{num}.jpg" for num in artificial])
hand_drawn = ["5b", "13a", "13b", "13c", "13d"]
hand_drawn = po.load_images([texture_path / f"fig{num}.jpg" for num in hand_drawn])

# Why not visualize them as well?
fig = po.imshow(natural, col_wrap=4, title=None)
fig.suptitle("Natural textures", y=1.05)
fig = po.imshow(artificial, col_wrap=4, title=None)
fig.suptitle("Artificial textures", y=1.05)
fig = po.imshow(hand_drawn, col_wrap=4, title=None)
fig.suptitle("Hand-drawn textures", y=1.05)
```

## Different initial image

We can also change the initial image and run metamer synthesis. What does it look like if our target is a texture, but our initial image is a face?
