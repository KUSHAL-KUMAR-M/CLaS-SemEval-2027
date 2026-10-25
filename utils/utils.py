import os


def get_test_questions(data_dir=None, k=70):
    """
    Load validation questions from language-specific FLORES files.

    Args:
        data_dir: Path to the validation directory.
                  Defaults to <project_root>/data/validation.
        k: Number of examples to load per language (default: 70).

    Returns:
        Dictionary mapping language codes to lists of questions.
    """
    if data_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, "data", "validation")

    test_questions = {}

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Validation directory not found: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue

        lang_code = filename[:-4]
        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                questions = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            questions = questions[:k]
            test_questions[lang_code] = questions
            print(f"Loaded {len(questions)} questions for {lang_code}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not test_questions:
        raise ValueError(f"No language files found in {data_dir}")

    print(f"\nSuccessfully loaded {len(test_questions)} languages")
    return test_questions
