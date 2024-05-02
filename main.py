import io
import base64
from fastapi import FastAPI, UploadFile
from PIL import Image
from snapcrop.hough_line_corner_detector import HoughLineCornerDetector
from snapcrop.page_extractor import PageExtractor
from snapcrop.processors import FastDenoiser, OtsuThresholder, Resizer, RotationCorrector
from snapner.ner import generate_tags
from snapocr.predict import recognize
from snapbase.database import Database
from fastapi import Form
import os
from datetime import datetime
import spacy

app = FastAPI()

database = Database()

nlp = spacy.load("en_core_web_sm")


page_extractor = PageExtractor(
        preprocessors = [
            Resizer(height = 2160, output_process = True), 
            # RotationCorrector(),
            FastDenoiser(strength = 5, output_process = True),
            OtsuThresholder(output_process = True),
        ],
        corner_detector = HoughLineCornerDetector(
            rho_acc = 1,
            theta_acc = 180,
            thresh = 100,
            output_process = True
        )
    )


@app.get("/")
def online():
    return {"I am": "online"}

@app.post("/snapservice")
async def snapservice(image_file: UploadFile):
    if image_file.filename.endswith(".jpg") or image_file.filename.endswith(".png") or image_file.filename.endswith(".jpeg"):
        img = Image.open(image_file.file)
        cropped_document = page_extractor(img)
        recognized_words = recognize(cropped_document)
        extracted_tags = generate_tags(nlp, recognized_words)

        # Convert cropped document to image readable format
        buffer = io.BytesIO()
        Image.fromarray(cropped_document).save(buffer, format="PNG")
        cropped_document_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Convert extracted tags to list of words
        return {"cropped_document": cropped_document_data, "extracted_tags": extracted_tags}
        
    return {"error": "File not supported"}

@app.get("/recentNotes")
async def recent_notes():
    recent_images = database.get_recent_images()
    recent_images_list = []
    for image in recent_images:
        image_data = Image.open(image[1])
        buffer = io.BytesIO()
        image_data.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_dict = {
            "image_id": image[0],
            "image_data": base64_image,
            "favorite": image[2]
        }
        recent_images_list.append(image_dict)
    return recent_images_list

@app.get("/favoriteNotes")
async def favorite_notes():
    favorite_images = database.get_favorite_images()
    favorite_images_list = []
    for image in favorite_images:
        image_data = Image.open(image[1])
        buffer = io.BytesIO()
        image_data.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_dict = {
            "image_id": image[0],
            "image_data": base64_image,
            "favorite": image[2]
        }
        favorite_images_list.append(image_dict)
    return favorite_images_list

@app.get("/searchNotes")
async def search_notes(tag: str):
    tags = tag.split(" ")
    searched_images = []
    for tag in tags:
        images = database.get_images(tag)
        for image in images:
            if image in searched_images:
                continue
            image_data = Image.open(image[1])
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_dict = {
                "image_id": image[0],
                "image_data": base64_image,
                "favorite": image[2]
            }
            searched_images.append(image_dict)
    return searched_images

@app.post("/uploadNote")
async def upload_note(image_file: UploadFile = Form(), tags: str = Form()):
    if image_file.filename.endswith(".jpg") or image_file.filename.endswith(".png") or image_file.filename.endswith(".jpeg"):
        contents = image_file.file.read()
        
        img_bytes = base64.b64decode(contents)
        img = Image.open(io.BytesIO(img_bytes))

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{image_file.filename}"
        filepath = os.path.join("notes", filename)

        img.save(filepath)

        image_id = database.insert_image(filepath, tags.split(" "))

        return {"image_id": image_id}


@app.post("/deleteNote")
async def delete_note(image_id: int = Form()):
    database.delete_image(image_id)
    return {"message": "Note deleted"}

@app.post("/flushTables")
async def drop_tables():
    database.drop_tables()
    database.create_tables()
    return {"message": "Tables dropped"}

@app.post("/setFavorite")
async def set_favorite(image_id: int = Form(), fav_val: bool = Form()):
    database.set_favorite(fav_val=fav_val, image_id=image_id)
    return {"message": "Favorite Changed"}