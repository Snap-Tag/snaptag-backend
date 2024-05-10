from torchvision import transforms
import torch

from snapocr.segmentation import segment_text
from .model import CNNModel
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '7', '8', '9']
character_classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}

def _predict_image(model, image):
    img_height = 28
    img_width = 28



    # Load and preprocess the image
    transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Grayscale()
    ])


    image = transform(image)
    image = image.unsqueeze(0)

    # Move the image to the GPU if available
    image = image.to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image)

    # Get the predicted class index and confidence
    predicted_class_idx = torch.argmax(predictions[0]).item()
    confidence = 100 * torch.max(predictions[0]).item()
    
    # Return the result
    result = {
        "predicted_class": character_classes[int(classes[int(predicted_class_idx)])],
        "confidence": confidence
    }
    return result


def recognize(image):
    '''
    Function to recognize the character.
    '''
    model = CNNModel(num_classes=62)
    model = model.to(device)
    model.load_state_dict(torch.load("snapocr/model.pth",map_location=device))
    model.eval()

    text_list = []
    segmented_image_list = segment_text(image)
    for lines in segmented_image_list:
        word_list = []
        for words in lines:
            char_list = []
            for characters in words:
                character = Image.fromarray(characters)
                recognized_character = _predict_image(model,character)
                if recognized_character["confidence"] > 85 and " " not in recognized_character["predicted_class"]:
                    char_list.append(recognized_character["predicted_class"])
            word_list.append("".join(char_list))
        text_list.append("".join(word_list))

    return " ".join(text_list)


if __name__ == "__main__":
    _image = Image.open("snapocr/test_images/13.png").convert("RGB")
    print(recognize(_image))
