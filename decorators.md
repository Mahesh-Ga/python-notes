Decorators
- A function that wraps another function/class to extend or modify its behavior without changing its source.
- Uses `@` syntax, executed at definition time.
- Typically returns a new callable that wraps the original.



## Types of decorators

### 1. Function decorators
Wrap a plain function to add behavior before/after the call (as above with `timer`).
```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Dividing {args[0]} by {args[1]}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@logger
def divide(a, b):
    return a / b

divide(10, 2)
# Output:
# Dividing 10 by 2
# Result: 5.0
```


### 2. Preserving metadata with `functools.wraps`
Without it, `wrapper.__name__`/`__doc__` shadow the original function's. Use `@wraps(func)` inside your decorator.
```python
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```