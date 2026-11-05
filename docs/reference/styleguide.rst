.. _styleguide_note:

Style Guide
================================

cuRobo follows the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.

In addition, the following principles are also followed:

- Avoid inheriting from configuration dataclasses and instead set ``MainClass.config`` during
  initialization.
- Avoid instantiating main classes inside a configuration dataclass.
- Use ``@get_torch_jit_decorator`` to decorate methods used in JIT-compiled functions.
- Make JIT functions part of the class using ``@staticmethod`` and ``@get_torch_jit_decorator``,
  and name them ``jit_<function_name>``.





Configuration Dataclasses
-------------------------
Configuration dataclasses group the settings for a class. Store the configuration on
``MainClass.config``. Initialize tensor-valued fields in ``__post_init__`` when they depend
on other settings, such as device or dtype.




