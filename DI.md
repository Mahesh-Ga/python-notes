# Dependency Injection (DI)

- A design pattern where an object's dependencies (the other objects/services it needs) are **provided from the outside** instead of the object creating them itself.
- Goal: **decouple** "what a class needs" from "how that thing is constructed."
- Answers: "Who hands me my dependencies?" instead of "How do I go build my own dependencies?"

## Without DI (tight coupling)
```python
class EmailService:
    def send(self, msg):
        print(f"Sending: {msg}")

class UserRegistration:
    def __init__(self):
        self.emailer = EmailService()   # hard-coded dependency

    def register(self, user):
        self.emailer.send(f"Welcome {user}")
```
- `UserRegistration` is stuck with `EmailService`. Can't swap it for a `SMSService` or a fake/mock in tests without editing the class.

## With DI (loose coupling)
```python
class UserRegistration:
    def __init__(self, notifier):     # dependency injected in
        self.notifier = notifier

    def register(self, user):
        self.notifier.send(f"Welcome {user}")

email = EmailService()
UserRegistration(email).register("Mahesh")   # inject EmailService

# testing: inject a fake, no real email sent
class FakeNotifier:
    def send(self, msg):
        print(f"[TEST] {msg}")

UserRegistration(FakeNotifier()).register("Mahesh")
```

## Types of Injection
- **Constructor Injection** — dependency passed into `__init__` (most common, shown above).
- **Setter Injection** — dependency set via a setter/attribute after construction.
    ```python
    ur = UserRegistration()
    ur.set_notifier(EmailService())
    ```
- **Interface/Method Injection** — dependency passed as a parameter to the method that needs it, not stored on the object.
    ```python
    def register(self, user, notifier):
        notifier.send(f"Welcome {user}")
    ```

## Why it matters
- **Testability** — swap real dependencies for mocks/fakes/stubs in unit tests (see fixtures in [[4. Testing]]).
- **Flexibility** — change implementations (Email → SMS → Push) without touching the consumer class.
- **Single Responsibility** — class focuses on its own logic, not on constructing collaborators.
- **Follows the Dependency Inversion Principle (the "D" in SOLID)** — depend on abstractions (interfaces), not concrete implementations.

## DI Container / Framework
- Manually injecting dependencies everywhere gets tedious in large apps → a **DI container** (IoC container) automates wiring: it knows how to construct objects and injects them wherever needed.
- Examples: `dependency-injector` (Python), Spring (Java), Angular's built-in DI, .NET's `IServiceCollection`.
- Basic idea: register a type/interface once, container resolves and injects instances wherever requested.
    ```python
    # dependency-injector style (conceptual)
    from dependency_injector import containers, providers

    class Container(containers.DeclarativeContainer):
        notifier = providers.Factory(EmailService)
        registration = providers.Factory(UserRegistration, notifier=notifier)

    container = Container()
    ur = container.registration()   # EmailService auto-injected
    ```

## Inversion of Control (IoC) vs DI
- **IoC** is the broader principle: control of flow/object creation is inverted — handed to a framework/container instead of the code itself (also seen in callbacks, event loops, templates).
- **DI** is one specific technique for achieving IoC, focused on supplying dependencies from outside.
