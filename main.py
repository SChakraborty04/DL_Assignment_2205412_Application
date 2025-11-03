import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

# Import your model classes from model.py
from model import VisionTransformer, SimpleCNN

# --- App & Template Setup ---
app = FastAPI(title="CIFAR-10 Dual Classifier")
templates = Jinja2Templates(directory="templates")

# --- Model & Preprocessing Setup ---

# Define the classes
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load CNN Model ---
CNN_MODEL_PATH = 'best_simple_cnn.pth'
cnn_model = SimpleCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.eval()

# --- Load ViT Model ---
# These parameters must match your notebook
VIT_MODEL_PATH = 'best_vision_transformer.pth'
vit_model = VisionTransformer(
    image_size=32,
    patch_size=8,
    num_classes=10,
    embed_dim=192,
    num_heads=6,
    mlp_dim=768,  # 192 * 4
    num_layers=6,
    dropout=0.1
).to(device)
vit_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
vit_model.eval()

# Define the image transformations
# IMPORTANT: Must match the test transform from your notebook + Resize
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize uploaded image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# --- End of Model Setup ---


def get_predictions(image_bytes):
    """
    Function to predict the class of an image using both models.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations and add batch dimension
        img_t = transform(image).to(device)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            # CNN Prediction
            cnn_out = cnn_model(batch_t)
            _, cnn_idx = torch.max(cnn_out.data, 1)
            cnn_pred = CLASSES[cnn_idx.item()]
            
            # ViT Prediction
            vit_out = vit_model(batch_t)
            _, vit_idx = torch.max(vit_out.data, 1)
            vit_pred = CLASSES[vit_idx.item()]
        
        return {
            "cnn_prediction": cnn_pred,
            "vit_prediction": vit_pred
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- API Routes ---

@app.get("/")
async def index(request: Request):
    """Render the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return JSON predictions."""
    if not file.content_type.startswith('image/'):
        return {"error": "File is not an image"}, 400

    try:
        img_bytes = await file.read()
        predictions = get_predictions(img_bytes)
        
        if predictions:
            return predictions
        else:
            return {"error": "Could not process image"}, 500
            
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 500