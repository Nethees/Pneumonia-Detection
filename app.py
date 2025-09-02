import streamlit as st
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
from Xray.entity.artifact_entity import ModelTrainerArtifact
# streamlit component
import pickle
import glob
from Xray.entity.config_entity import ModelTrainerConfig


class Application:
    
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_trainer_config = model_trainer_config

    # this is for saving images and prediction
    def save_image(self, uploaded_file):
        if uploaded_file is not None:
            # save the uploaded image
            os.makedirs("images", exist_ok=True)
            image_path = os.path.join("images", "input.jpeg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Image saved to {image_path}")

            #model = torch.load(Path('model/model.pt'))
            # Load model using artifact path
            model_path = self.model_trainer_artifact.trained_model_path
            model = torch.load(model_path, map_location=torch.device("cpu"))
        

            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                ])

            image = Image.open(image_path).convert("RGB")

            input = trans(image)

            
            #input = input.view(1, 1, 224, 224).repeat(1, 3, 1, 1)
            input = input.unsqueeze(0)  # Shape: [1, 3, 224, 224]

            output = model(input)

            prediction = int(torch.max(output.data, 1)[1].numpy())
            print(prediction)

            if (prediction == 0):
                print ('Normal')
                st.text_area(label="Prediction:", value="Normal", height=100)
            if (prediction == 1):
                print ('PNEUMONIA')
                st.text_area(label="Prediction:", value="PNEUMONIA", height=100)


if __name__ == "__main__":
    st.title("Xray lung classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    config = ModelTrainerConfig()
    try:
        # Load latest model.pkl
        latest_model_dir = max(glob.glob("artifacts/*/model_training"), key=os.path.getmtime)
        model_pkl_path = os.path.join(latest_model_dir, "model.pkl")

        with open(model_pkl_path, "rb") as f:
            model_trainer_artifact: ModelTrainerArtifact = pickle.load(f)

        config = ModelTrainerConfig()
        ap = Application(model_trainer_artifact=model_trainer_artifact, model_trainer_config=config)
        ap.save_image(uploaded_file)

    except Exception as e:
        st.error(f"Error: {e}")

