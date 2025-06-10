Model and Datasets
==================

DeepCell models and training datasets are licensed under a 
[modified Apache license][license] for non-commercial academic use only.
An API key for accessing datasets and models can be obtained at <https://users.deepcell.org/login/>.

[license]: https://github.com/vanvalenlab/deepcell-types/blob/master/LICENSE

API Key Usage
-------------

The token that is issued by <https://users.deepcell.org> should be added as an
environment variable:

```bash
export DEEPCELL_ACCESS_TOKEN=<token-from-users.deepcell.org>
```

This line can be added to your shell configuration (e.g. ``.bashrc``, ``.zshrc``,
``.bash_profile``, etc.) to automatically grant access to DeepCell models/data
upon login.

(download_models)=
Models
------

The model can be downloaded for local use:

```python
>>> from utils import download_model


>>> download_model()
```

Training Data
-------------

```{warning}
The training dataset is over 1.3 TB - make sure you have space and sufficient
network bandwidth before attempting to download.
```

Similarly, training data can be downloaded for local use with:

```python
>>> from utils import download_training_data


>>> download_training_data()
```
