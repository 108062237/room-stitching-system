class AssemblyError(ValueError):
    """Documented input/availability error; suitable for an HTTP adapter."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}

