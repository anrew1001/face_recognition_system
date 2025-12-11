import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from insightface.app import FaceAnalysis
from database import IdentityDatabase
from PIL import Image
import cv2
import numpy as np
from recognition.types import EmbeddingResult, FaceDetection


def build_from_known(known_dir: str, output_path: str):
    known_path = Path(known_dir)

    # InsightFace уже загружен при prepare()
    model = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    model.prepare(ctx_id=-1, det_size=(640, 640))

    db = IdentityDatabase(max_embeddings_per_identity=5)

    for person_dir in known_path.iterdir():
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        for img_path in images[:5]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # InsightFace get() возвращает список лиц
            faces = model.get(img)
            if not faces:
                continue

            face = faces[0]  # Первое лицо

            # Создать EmbeddingResult вручную
            emb = face.embedding / np.linalg.norm(face.embedding)

            x1, y1, x2, y2 = [int(x) for x in face.bbox]
            crop = img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            result = EmbeddingResult(
                embedding=emb.astype(np.float32),
                face_crop=crop_pil,
                detection=FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(face.det_score)
                ),
                model_fingerprint="insightface:buffalo_l:512"
            )

            db.add_embedding_result(name, result)
            print(f"Added {name}: {img_path.name}")

    db.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent  # корень проекта
    known_dir = script_dir / "known"
    output_path = script_dir / "data" / "identities.npz"
    build_from_known(str(known_dir), str(output_path))