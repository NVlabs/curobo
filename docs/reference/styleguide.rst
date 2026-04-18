.. _styleguide_note:

Style Guide
================================

cuRobo follows the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_.

In addition, the following principles are also followed:

- Avoid inheriting from configuration dataclasses and instead set MainClass.config during
initialization.
- Avoid instantiaing main classes inside a configuration dataclass.
- Use `@get_torch_jit_decorator` to decorate methods that are used in JIT compiled functions.
- Make jit functions be part of the class using `@staticmethod` and `@get_torch_jit_decorator`, and
  name them as `jit_<function_name>`.





Dataclass as Configuration Class:
-----------------------------------
Many classes in cuRobo use dataclasses to store configuration parameters. Current implementation
moves the parameters to torch tensors in the `__post_init__` method. The alternative is to
initialize the torch tensors as private variables in the dataclass. We are leaving this for a
future PR as we are unsure of the tradeoffs.




