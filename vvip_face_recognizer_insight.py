import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from thumbor.utils import logger

class VVIPFaceRecognizerInsight:
    def __init__(self, vvip_faces_dir="vvip_faces", models_dir="models", tolerance=1.0):
        """
        tolerance=
        0.3 = strict match
        0.4–0.5 = common
        0.6+ = may cause false positives
        """
        self.vvip_faces_dir = vvip_faces_dir
        self.models_dir = models_dir
        self.tolerance = tolerance
        self.model_path = os.path.join(self.models_dir, "vvip_insight_encodings.pkl")

        self.known_face_embeddings = []
        self.known_face_names = []

        self.face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0)

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.known_face_embeddings = model_data['embeddings']
                self.known_face_names = model_data['names']
                print(f"✓ Loaded {len(self.known_face_names)} VVIP faces (InsightFace)")
                return True
            except Exception as e:
                logger.error(f"Error loading InsightFace model: {e}")
        return False

    def train_model(self):
        if not os.path.exists(self.vvip_faces_dir):
            logger.error(f"VVIP faces directory not found: {self.vvip_faces_dir}")
            return False

        self.known_face_embeddings = []
        self.known_face_names = []

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        for person_name in os.listdir(self.vvip_faces_dir):
            person_dir = os.path.join(self.vvip_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            print(f"Training on VVIP: {person_name}")
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(person_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    faces = self.face_analyzer.get(img)
                    if len(faces) == 0:
                        print(f"  No faces found in {img_file}, skipping")
                        continue

                    # Use the most prominent face
                    emb = faces[0].embedding
                    self.known_face_embeddings.append(emb)
                    self.known_face_names.append(person_name)
                    print(f"  Added embedding from {img_file}")
                except Exception as e:
                    print(f"  Error processing {img_file}: {e}")

        if len(self.known_face_embeddings) > 0:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.known_face_embeddings,
                    'names': self.known_face_names
                }, f)
            print(f"✓ Saved InsightFace model with {len(self.known_face_names)} embeddings")
            return True
        else:
            print("✗ No embeddings found")
            return False

    def recognize_faces(self, image):
        results = []
        faces = self.face_analyzer.get(image)

        for face in faces:
            emb = face.embedding

            # Normalize the embedding
            emb_norm = emb / np.linalg.norm(emb)
            known_norms = np.array([vec / np.linalg.norm(vec) for vec in self.known_face_embeddings])

            # Compute cosine similarity
            sims = np.dot(known_norms, emb_norm)
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            name = "Unknown"
            if best_score >= self.tolerance:  # e.g. 0.35 is a decent starting threshold
                name = self.known_face_names[best_idx]

                # Get face box
                box = face.bbox.astype(int)
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                results.append((x, y, w, h, name, best_score))

                logger.info(f"Detected VVIP: {name}, similarity: {best_score:.3f}")

        return results

