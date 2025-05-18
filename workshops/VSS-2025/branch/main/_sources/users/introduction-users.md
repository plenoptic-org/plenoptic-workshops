---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: plenoptic_venv
  language: python
  name: plenoptic_venv
---
# Introduction
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/introduction.md)


<img src="../_static/models.png">

For the purposes of this notebook, we'll use some very simple convolutional models that are inspired by the processing done in the lateral geniculate nucleus (LGN) of the visual system[^models]. We're going to build up in complexity, starting with the Gaussian model at the top and gradually adding features[^notallmodels]. We'll describe the components of these models in more detail as we get to them, but briefly:

[^models]: Most of these models were originally published in Berardino, A., Laparra, V., J Ball\'e, & Simoncelli, E. P. (2017). Eigen-distortions of hierarchical representations. In Adv. Neural Information Processing Systems (NIPS*17), from which the figure is modified.

[^notallmodels]: Note that the Berardino et. al, 2017 paper includes more models than described here. We're not examining all of them for time's sake, but you can check out the rest of the models described in the Berardino paper, they're all included in plenoptic under the [plenoptic.simulate.FrontEnd](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.simulate.models.html#module-plenoptic.simulate.models.frontend) module!

- `Gaussian`: the model just convolves a Gaussian with an image, so that the model's representation is simply a blurry version of the image.
- `CenterSurround`: the model convolves a difference-of-Gaussian filter with the image, so that model's representation is bandpass, caring mainly about frequencies that are neither too high or too low.
- `LuminanceGainControl`: the model rectifies and normalizes the linear component of the response using a local measure of luminance, so that the response is invariant to local changes in luminance.
## Plenoptic basics
```{code-cell} ipython3
:tags: [render-all]

import plenoptic as po
import torch
import pyrtools as pt
import matplotlib.pyplot as plt
# so that relative sizes of axes created by po.imshow and others look right
plt.rcParams['figure.dpi'] = 72
plt.rcParams['animation.html'] = 'html5'
# use single-threaded ffmpeg for animation writer
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']

%matplotlib inline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print("Running on GPU!")
else:
    print("Running on CPU!")
# for reprodicibility
po.tools.set_seed(1)

def plot_helper(metamer, init_img=None):
    if init_img is None:
        init_img = metamer.saved_metamer[0]
    if metamer.image.shape[0] > 1:
        img = metamer.image[:1]
    else:
        img = metamer.image
    to_plot = [torch.cat([torch.ones_like(img),
                          img,
                          metamer.model(img)])]
    for i, j in zip(init_img, metamer.metamer):
        to_plot.append(torch.stack([i, j, metamer.model(j)]))
    to_plot = torch.cat(to_plot)
    fig = po.imshow(to_plot, col_wrap=3,
                    title=['', 'Original image', 'Model representation\nof original image']+
                           3*['Initial image', 'Synthesized metamer', 'Model representation\nof synthesized metamer']);
    # change the color scale of the images so that the first two columns go from 0 to 1
    # and the last one is consistent
    for ax in fig.axes:
        if 'representation' in ax.get_title():
            clim = (to_plot[2::3].min(), to_plot[2::3].max())
        else:
            clim = (0, 1)
        ax.images[0].set_clim(*clim)
        title = ax.get_title().split('\n')
        title[-2] = f" range: [{clim[0]:.01e}, {clim[1]:.01e}]"
        ax.set_title('\n'.join(title))
    return fig

```


All synthesis methods require a "reference" or "target" image, so let's load one in.
```{code-cell}
# enter code here
```



Set up the Guassian model. Models in plenoptic must:
- Inherit `torch.nn.Module`.
- Have `forward` and `__init__` methods.
- Accept tensors as input and return tensors as output.
- All operations performed must be torch-differentiable (i.e., come from the torch library)
- Have all model parameter gradients removed.
```{code-cell} ipython3
:tags: [render-all]

# this is a convenience function for creating a simple Gaussian kernel
from plenoptic.simulate.canonical_computations.filters import circular_gaussian2d

# Simple Gaussian convolutional model
class Gaussian(torch.nn.Module):
    # in __init__, we create the object, initializing the convolutional weights and nonlinearity
    def __init__(self, kernel_size, std_dev=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(0, 0), bias=False)
        self.conv.weight.data[0, 0] = circular_gaussian2d(kernel_size, std_dev)
        
    # the forward pass of the model defines how to get from an image to the representation
    def forward(self, x):
        x = po.tools.conv.same_padding(x, self.kernel_size, pad_mode='circular')
        return self.conv(x)
```


- Initialize the Gaussian model.
- Call it on our image.
- View the dimensionality of the model inputs and outputs.
```{code-cell}
# enter code here
```



The following shows the image and the model output. We can see that output is a blurred version of the input, as we would expect from a low-pass model.

- The Gaussian model output is a blurred version of the input.
- This is because the model is preserving the low frequencies,  discarding the high frequencies (i.e., it's a lowpass filter).
- Thus, this model is completely insensitive to high frequencies -- information there is invisible to the model.
```{code-cell}
# enter code here
```

## Examining model invariances with metamers


- Initialize the `Metamer` object and synthesize a model metamer.
```{code-cell}
# enter code here
```



- View the synthesis process.
```{code-cell}
# enter code here
```



:::{important} 
This next cell will take a while to run --- making animations in matplotlib is a bit of a slow process.
:::
```{code-cell}
# enter code here
```



- Visualize model metamers.
```{code-cell}
# enter code here
```



- Synthesize more model metamers, from different starting points.
```{code-cell} ipython3
:tags: [render-all]

curie = po.data.curie().to(DEVICE)
# pyrtools, imported as pt, has a convenience function for generating samples of white noise, but then we still 
# need to do some annoying things to get it ready for plenoptic
pink = torch.from_numpy(pt.synthetic_images.pink_noise((256, 256))).unsqueeze(0).unsqueeze(0)
pink = po.tools.rescale(pink).to(torch.float32).to(DEVICE)
po.imshow([curie, pink]);
```


Visualize all metamer outputs. In the plot we will create:
- the first row shows our target Einstein image and its model representation, as we saw before.
- the new three rows show our model metamers resulting from three different starting points.
- in each, the first column shows the starting point of our metamer synthesis, the middle shows the resulting model metamer, and the third shows the model representation.


We can see that the model representation is the same for all four images, but the images themselves look very different. Because the model is completely invariant to high frequencies, the high frequencies present in the initial image are not affected by the synthesis procedure and thus are still present in the model metamer.
```{code-cell} ipython3
:tags: [render-all]

fig = po.imshow([torch.ones_like(img), img, rep,
                 metamer.saved_metamer[0], metamer.metamer, model(metamer.metamer),
                 pink, metamer_pink.metamer, model(metamer_pink.metamer),
                 curie, metamer_curie.metamer, model(metamer_curie.metamer)],
                col_wrap=3, vrange='auto1',
                title=['', 'Original image', 'Model representation\nof original image']+
                      3*['Initial image', 'Synthesized metamer', 'Model representation\nof synthesized metamer']);
```
## Examining model sensitivies to eigendistortions


- While metamers allow us to examine model invariances, eigendistortions allow us to also examine model sensitivities.
- Eigendistortions are distortions that the model thinks are the most and least noticeable.
```{code-cell}
# enter code here
```

## A more complex model


- The `CenterSurround` model has bandpass sensitivity, as opposed to the `Gaussian`'s lowpass.
- Thus, it is still insensitive to the highest frequencies, but it is less sensitive to the low frequencies the Gaussian prefers, with its peak sensitivity lying in a middling range.
```{code-cell}
# enter code here
```



- We can synthesize all three model metamers at once by taking advantage of multi-batch processing.
```{code-cell}
# enter code here
```



- Visualize all the model metamers we synthesized.
```{code-cell}
# enter code here
```



- By examining the eigendistortions, we can see more clearly that the model's preferred frequency has shifted higher, while the minimal eigendistortion still looks fairly similar.
```{code-cell}
# enter code here
```

## Adding some nonlinear features to the mix


- The `LuminanceGainControl` model adds a nonlinearity, gain control. This makes the model harder to reason than the first two models.
- This model divides the output of the `CenterSurround` filter with an estimate of local luminance (the output of a larger Gaussian filter), which makes the model completely insensitive to absolute pixel values. It now cares about contrast, rather than luminance.
- This is a computation that we think is present throughout much of the early visual system.
```{code-cell}
# enter code here
```



- Let's synthesize and visualize some metamers for this model.
```{code-cell}
# enter code here
```



- Now let's use eigendistortions to see what this model is particularly sensitive to.
```{code-cell}
# enter code here
```

## Conclusion


In this notebook, we saw the basics of using `plenoptic` to investigate the sensitivities and invariances of some simple convolutional models, and reasoned through how the model metamers and eigendistortions we saw enable us to understand how these models process images.

`plenoptic` includes a variety of models and model components in the [plenoptic.simulate](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.simulate.html) module, and you can (and should!) use the synthesis methods with your own models. Our documentation also has [examples](https://docs.plenoptic.org/docs/branch/main/tutorials/applications/Demo_Eigendistortion.html) showing how to use models from [torchvision](https://pytorch.org/vision/stable/index.html) (which contains a variety of pretrained neural network models) with plenoptic (we'll be releasing a simpler interface for torchvision and other pytorch model zoos this summer -- ask me if you're interested!). In order to use your own models with plenoptic, check the [documentation](https://docs.plenoptic.org/docs/branch/main/models.html) for the specific requirements, and use the [`validate_model`](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.tools.html#plenoptic.tools.validate.validate_model) function to check compatibility. If you have issues or want feedback, we're happy to help --- just post on the [Github discussions page](https://github.com/plenoptic-org/plenoptic/discussions)!