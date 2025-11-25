from typing import List, Optional

class ProjectFile:
    def __init__(self, path: str, content: str):
        self.path = path
        self.content = content

class Project:
    def __init__(self, files: List[ProjectFile], summary: Optional[str] = None):
        self.files = files
        self.summary = summary

