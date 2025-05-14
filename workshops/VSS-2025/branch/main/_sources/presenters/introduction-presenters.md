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
```
```{code-cell} ipython3
lg_curie_eig = po.synthesize.Eigendistortion(curie, lg)
lg_curie_eig.synthesize();
po.imshow(lg_curie_eig.eigendistortions, title=['LG Maximum eigendistortion\n (on Curie)', 
                                                'LG Minimum eigendistortion\n (on Curie)']);
cs_curie_eig = po.synthesize.Eigendistortion(curie, center_surround)
cs_curie_eig.synthesize();
po.imshow(cs_curie_eig.eigendistortions, title=['CenterSurround Maximum \neigendistortion (on Curie)', 
                                                'CenterSurround Minimum \neigendistortion (on Curie)']);
po.imshow(cs_eig.eigendistortions, title=['CenterSurround Maximum \neigendistortion (on Einstein)', 
                                          'CenterSurround Minimum \neigendistortion (on Einstein)']);
```
```{code-cell} ipython3
# the [:1] is a trick to get only the first element while still being a 4d
# tensor
po.imshow([img+3*lg_eig.eigendistortions[:1],
           curie+3*lg_curie_eig.eigendistortions[:1]]);
```
```{code-cell} ipython3
po.imshow(img+3*lg_eig.eigendistortions[:1].roll(128, -1))
print(f"Max LG eigendistortion: {po.tools.l2_norm(lg(img), lg(img+lg_eig.eigendistortions[:1]))}")
print(f"Shifted max LG eigendistortion: {po.tools.l2_norm(lg(img), lg(img+lg_eig.eigendistortions[:1].roll(128, -1)))}")
```
```{code-cell} ipython3
print(f"Max CenterSurround eigendistortion: {po.tools.l2_norm(center_surround(img), center_surround(img+cs_eig.eigendistortions[:1]))}")
print(f"Shifted max CenterSurround eigendistortion: {po.tools.l2_norm(center_surround(img), center_surround(img+cs_eig.eigendistortions[:1].roll(128, -1)))}")
```
## Conclusion


In this notebook, we saw the basics of using `plenoptic` to investigate the sensitivities and invariances of some simple convolutional models, and reasoned through how the model metamers and eigendistortions we saw enable us to understand how these models process images.

`plenoptic` includes a variety of models and model components in the [plenoptic.simulate](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.simulate.html) module, and you can (and should!) use the synthesis methods with your own models. Our documentation also has [examples](https://docs.plenoptic.org/docs/branch/main/tutorials/applications/Demo_Eigendistortion.html) showing how to use models from [torchvision](https://pytorch.org/vision/stable/index.html) (which contains a variety of pretrained neural network models) with plenoptic (we'll be releasing a simpler interface for torchvision and other pytorch model zoos this summer -- ask me if you're interested!). In order to use your own models with plenoptic, check the [documentation](https://docs.plenoptic.org/docs/branch/main/models.html) for the specific requirements, and use the [`validate_model`](https://docs.plenoptic.org/docs/branch/main/api/plenoptic.tools.html#plenoptic.tools.validate.validate_model) function to check compatibility. If you have issues or want feedback, we're happy to help --- just post on the [Github discussions page](https://github.com/plenoptic-org/plenoptic/discussions)!

