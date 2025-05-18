---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Minimal metamer synthesis example

See [plenoptic docs](https://docs.plenoptic.org/) for more details.

```{code-cell} ipython3
import plenoptic as po
import torch
# needed for the plotting/animating:
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['animation.html'] = 'html5'
# use single-threaded ffmpeg for animation writer
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The following code block:
- initialize an image and a model
- run metamer synthesis until convergence

```{code-cell} ipython3
img = po.data.einstein().to(DEVICE)
model = po.simul.LuminanceGainControl(
    kernel_size=(31, 31), pad_mode="circular",
    pretrained=True, cache_filt=True
)
model.to(DEVICE)
po.tools.remove_grad(model)
model.eval()
met = po.synth.Metamer(img, model)
met.synthesize(max_iter=1300, stop_criterion=1e-11, store_progress=10)
```

Next, we need to ensure that the metamer synthesis succeeded. In the previous example we worked through, I had ensured that we ran the synthesis for long enough that we didn't need to check this, but in general, you do.

There are many possible visualizations one can make. We have a helper function that should help get you started. It shows the metamer, the synthesis loss over time, and (if possible) the representation error.

```{code-cell} ipython3
po.synth.metamer.plot_synthesis_status(met);
```

In the above figure, we can see that the loss has decreased to a low value and, importantly, that it looks like it has stabilized.

The representation error is easier to understand if we view it over time, which we can do with the following helper function:

```{code-cell} ipython3
po.synth.metamer.animate(met)
```

We can see that the representation error decreases relatively uniformly across the image.

## Different target image

Try using a different target image than the one of Einstein above and running metamer synthesis until completion:

:::{admonition} Loading other images
:class: hint

Try one of the other [included images](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.data.html#module-plenoptic.data) or use [`load_images`](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.tools.html#plenoptic.tools.data.load_images) to load one from disk.

:::

```{code-cell} ipython3
:tags: [skip-execution]

img = # WRITE SOMETHING NEW HERE
img = img.to(DEVICE)
met = po.synth.Metamer(img, model)
met.synthesize(max_iter=1300, stop_criterion=1e-11, store_progress=10)
po.synth.metamer.plot_synthesis_status(met);
```

And maybe animate to see what synthesis looks like?

```{code-cell} ipython3
:tags: [skip-execution]

po.synth.metamer.animate(met)
```

## Different initial image

While we often initialize from a patch of white noise, it can be interesting to start from a different image as well. Using one of the same tools as above for loading another image, initialize metamer synthesis from another starting point and run it to completion:

```{code-cell} ipython3
:tags: [skip-execution]

met = po.synth.Metamer(img, model)
met.setup(initial_image=) # FINISH THE CALL TO setup
met.synthesize(max_iter=1300, stop_criterion=1e-11, store_progress=10)
po.synth.metamer.plot_synthesis_status(met);
```

And maybe animate to see what synthesis looks like?

```{code-cell} ipython3
:tags: [skip-execution]

po.synth.metamer.animate(met)
```

## Other models

Try any of the above with a different model! Try one of the other models from the [`frontend`](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.simulate.models.html#module-plenoptic.simulate.models.frontend) module.

If you want a more complex model, see the [texture](./textures.md) or [torchvision](./torchvision.md) notebooks.
