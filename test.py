import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer
def greet(name):
    print("Here")
    return f"Hello, {name}!"

greet("World")  # prints elapsed time, then returns "Hello, World!"