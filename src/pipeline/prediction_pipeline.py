import os
import sys
import pandas as pd
from pathlib import Path
from joblib import load
import gdown
import zipfile
import torch # Import torch to fix the device issue

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        try:
            # Check if artifacts exist, if not, download them
            self.ensure_artifacts_exist()

            # --- FIX FOR MPS/GPU ARTIFACTS ON CPU ---
            # This allows models trained on Mac (MPS) or GPU to load on Streamlit Cloud (CPU)
            # We temporarily patch torch.storage.UntypedStorage to force CPU mapping
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                logging.info("Detected CPU-only environment. Patching torch for artifact loading...")
                
                # Save the original class to restore later
                original_storage = torch.UntypedStorage
                
                # Define a patched class that ignores the 'device' argument
                class CPUMappedStorage(torch.UntypedStorage):
                    def __new__(cls, *args, **kwargs):
                        kwargs.pop('device', None) # Remove 'device' if present
                        return super().__new__(cls, *args, **kwargs)
                
                # Apply the patch
                torch.UntypedStorage = CPUMappedStorage
            
            # Load artifacts (now safe to load MPS/CUDA pickles)
            self.embedder = load(os.path.join("artifacts", "embedder.pkl"))
            self.model = load(os.path.join("artifacts", "model_lgbm.pkl"))
            self.le = load(os.path.join("artifacts", "label_encoder.pkl"))
            
            # Restore original class after loading to avoid side effects
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                 torch.UntypedStorage = original_storage

            logging.info("Loaded embedder, model, and label encoder artifacts.")
        except Exception as e:
            raise CustomException(e, sys)

    def ensure_artifacts_exist(self):
        """
        Checks if artifacts exist. If not, downloads them from Google Drive.
        """
        artifact_path = "artifacts/embedder.pkl" # Check for a key file
        
        if not os.path.exists(artifact_path):
            logging.info("Artifacts not found locally. Downloading from Google Drive...")
            
            # --- REPLACE THIS ID WITH YOUR ACTUAL GOOGLE DRIVE FILE ID ---
            # Example: If link is https://drive.google.com/file/d/1XyZ.../view
            # The ID is the part between /d/ and /view
            file_id = '1YKnwEV3iphpoyFeCe2Ku9WGduxOSftkD' 
            # -------------------------------------------------------------
            
            url = f'https://drive.google.com/uc?id={file_id}'
            output = 'artifacts.zip'
            
            try:
                # Download the zip file
                gdown.download(url, output, quiet=False)
                
                # Extract the zip file
                logging.info("Extracting artifacts.zip...")
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(".") # Extracts to current directory (should create/fill 'artifacts' folder)
                
                # Clean up zip file
                os.remove(output)
                logging.info("Artifacts downloaded and extracted successfully.")
                
            except Exception as e:
                logging.error(f"Failed to download artifacts: {e}")
                raise CustomException("Failed to download model artifacts from Google Drive.", sys)
        else:
            logging.info("Artifacts found locally.")

    def predict(self, raw_contract_text: str):
        try:
            # Embed the new text
            embeddings = self.embedder.encode([raw_contract_text])

            # Predict risk class index
            pred_idx = self.model.predict(embeddings)
            
            # Convert back to original label
            pred_label = self.le.inverse_transform(pred_idx)[0]

            result = {"risk_level": pred_label, "suggestion_prompt": None}
            if pred_label in ['high', 'medium']:
                result["suggestion_prompt"] = (
                    "You are an expert legal analyst. Review the following contract clause, "
                    "identify the primary risks, and suggest a more balanced and safer alternative. "
                    "Present your answer clearly with a 'Risk Analysis' section and a 'Suggested Rewrite' section.\n\n"
                    f"Clause to review: \"{raw_contract_text}\""
                )
            else:
                 result["suggestion_prompt"] = "No improvement suggestions needed for a low-risk clause."
            return result

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = PredictionPipeline()
    # Test with a dummy string
    print(pipeline.predict("This agreement shall be governed by the laws of New York."))
