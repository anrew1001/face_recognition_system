import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
from pathlib import Path
import cv2
from recognition import registry
from database import IdentityDatabase


def build_from_known(known_dir: str, output_path: str):
    """Загрузить БД из known/."""
    known_path = Path(known_dir)
    model = registry.get("scrfd_10g")
    model.load()

    db = IdentityDatabase(max_embeddings_per_identity=5)

    for person_dir in known_path.iterdir():
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        for img_path in images[:5]:  # Максимум 5
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            result = model.extract_embedding(img)
            if result:
                db.add_embedding_result(name, result)
                print(f"Added {name}: {img_path.name}")

    db.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    build_from_known("known", "data/identities.npz")