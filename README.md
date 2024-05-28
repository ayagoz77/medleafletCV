repo for detecting text and processing prompts related with detected text with llama3 api
-
![image](https://github.com/ayagoz77/medleafletCV/assets/72253810/851a3fb6-43c9-473a-a0f6-3934b70d0ce3)
---
TesseractOCR is used for OCR step and Layout Analysis step is simplified for demo version and looks like:

![image](https://github.com/ayagoz77/medleafletCV/assets/72253810/1debbbf7-25d8-446d-86e3-23a78ad6cc97)

Install requirements to your env.
```bash
pip install -r requirements.txt
```
Create .env file and set variables GROQ_API_KEY(groq cloud api key) and TOKEN(bot father from telegram)
and simply run
```bash
python tg_bot.py
```
---
Use text_handler.py with your cfg.yaml
```bash
python text_handler.py --cfg your_cfg.yaml
```
Each folder explanation for further unit testing:
```tree
├── block_extractor               
│   ├── block_id_y.py           - need to improve (block clutering after getting columns) not used
│   └── density_clustering_x.py - clustering into by x values
├── detectors
│   ├── corner_detector.py      - one point detector
│   ├── detector.py             - abstract class
│   ├── document_detector.py    - 4 point detector
│   ├── __init__.py
│   ├── model_weights           - passed as default
│   │   ├── cornerDetector.onnx
│   │   └── docDetector.onnx
│   └── test
│       └── quality_check.py    - calculates mean IoU (needs input as image.jpg and its label as image.jpg.csv with 4 points coordinates(order doesnt matter))
└── utils
    ├── __init__.py
    ├── quality_enhancer.py     - brightness and contrast enhancer
    └── text_extractor.py       - morphology operator for text bboxes extracting
```
Further improvements:
- Replace OCR model
- Try integrate ViT for direct processing
- Better prompt engineering
