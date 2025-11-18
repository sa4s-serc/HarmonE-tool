import os

# You can override this with an environment variable if needed
KNOWLEDGE_DIR = os.getenv("HARMONE_KNOWLEDGE_PATH", "knowledge")

def get_knowledge_file(filename):
    """
    Returns the absolute path to a file inside the knowledge directory.
    """
    return os.path.join(KNOWLEDGE_DIR, filename)
