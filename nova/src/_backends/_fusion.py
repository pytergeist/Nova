class FusionBackend:
    def __init__(self):
        self.fusion = None
        try:
            import fusion
            self.fusion = True
        except ImportError:
            raise ImportError("Fusion backend not found. Please install the fusion package.")


