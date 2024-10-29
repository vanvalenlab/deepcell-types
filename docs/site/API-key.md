DeepCell API Key
================

DeepCell models and training datasets are licensed under a 
[modified Apache license][license] for non-commercial academic use only.
An API key for accessing datasets and models can be obtained at <https://users.deepcell.org/login/>.

API Key Usage
-------------

The token that is issued by <https://users.deepcell.org> should be added as an
environment variable::

```bash
export DEEPCELL_ACCESS_TOKEN=<token-from-users.deepcell.org>
```

This line can be added to your shell configuration (e.g. ``.bashrc``, ``.zshrc``,
``.bash_profile``, etc.) to automatically grant access to DeepCell models/data
upon login.
