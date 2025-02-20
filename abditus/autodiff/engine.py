class Engine:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __del__(self):
        print('Engine deleted')

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def current(self):
        return self


if __name__ == "__main__":
    import time
    with Engine() as engine:
        print(engine)
        time.sleep(5)
        print(engine.current())
