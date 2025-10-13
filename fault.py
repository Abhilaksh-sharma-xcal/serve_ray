import ray

@ray.remote
class Actor:
    def read_only(self):
        import sys
        import random

        rand = random.random()
        if rand < 0.2:
            return 2 / 0
        elif rand < 0.3:
            sys.exit(1)

        return 2


actor = Actor.remote()
# Manually retry the actor task.
while True:
    try:
        print(f"das {ray.get(actor.read_only.remote())}")
        break
    except ZeroDivisionError:
        print("dasd")
        pass
    except ray.exceptions.RayActorError:
        print("dsdassasa")
        # Manually restart the actor
        actor = Actor.remote()